"""Mock OpenAI-compatible chat server for testing framework workers WITHOUT a GPU.
Policy: if the request has `tools`, return a tool_call to the first tool whose name contains a
candidate-like token (transfer_to_*, handoff_to_*, agent_*), else the first tool; if no tools,
return a text/JSON answer naming the first `agent_XXXXXX` mentioned in the prompt (also as
{"next_speaker": ..., "assignee_id": ..., "choice": 1 (1-based, LlamaIndex), "index": 0}). Usage: python scripts/mock_openai_server.py 8123
"""
import json, re, sys, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

AGENT = re.compile(r"agent_\d{6}")
TASK_ID = re.compile(r"Task ID: (\S+)")   # CAMEL Workforce ASSIGN_TASK_PROMPT enumerates tasks this way


def _schema(req):
    rf = req.get("response_format") or {}
    return (rf.get("json_schema") or {}).get("schema") or rf.get("schema")


AGENTISH = ("agent", "speaker", "assignee", "name", "select", "worker")


def _deref(spec, root):
    """Follow $ref / single-entry allOf, as pydantic emits for nested models."""
    for _ in range(8):
        if "$ref" in spec:
            spec = root.get("$defs", {}).get(spec["$ref"].rsplit("/", 1)[-1], {})
        elif len(spec.get("allOf") or []) == 1:
            spec = spec["allOf"][0]
        else:
            return spec
    return spec


def _fill(schema, first, root=None, agentish=False):
    """Minimal instance of a JSON schema: an agent-ish string field gets `first`, everything else a stub.
    Nested models are resolved and filled, and agent-ish-ness is inherited from the enclosing field, so a
    wrapper like Magentic-One's `next_speaker: {reason, answer}` still puts the agent name in `answer`."""
    root = schema if root is None else root
    out = {}
    for name, spec in (schema.get("properties") or {}).items():
        spec = _deref(spec, root)
        mine = agentish or any(w in name for w in AGENTISH)
        t = spec.get("type")
        t = next((x for x in t if x != "null"), "string") if isinstance(t, list) else t
        if t == "boolean": out[name] = False
        elif t in ("integer", "number"): out[name] = 0
        elif t == "array": out[name] = []
        elif t == "object" or spec.get("properties"): out[name] = _fill(spec, first, root, mine)
        elif mine: out[name] = first
        else: out[name] = "mock"
    return out


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        body = {"object": "list", "data": [{"id": "Qwen/Qwen2.5-7B-Instruct", "object": "model"}]} if "models" in self.path else {"ok": True}
        self._send(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); req = json.loads(self.rfile.read(n) or b"{}")
        # the roster lives in the messages for prompt-based frameworks and in a tool description for
        # tool-based ones (CrewAI lists coworkers there), so fall back to the whole request.
        names = AGENT.findall(json.dumps(req.get("messages", []))) or AGENT.findall(json.dumps(req))
        first = names[0] if names else "agent_000000"
        tools = req.get("tools") or []
        msg = {"role": "assistant", "content": None}
        if tools:
            pick = next((t for t in tools if AGENT.search(json.dumps(t))), tools[0])
            fn = pick["function"]["name"]
            props = pick["function"].get("parameters", {}).get("properties", {})
            args = {}
            for p in props:
                if "agent" in p or "coworker" in p or "name" in p or "assignee" in p:
                    args[p] = first
                elif "task" in p or "query" in p or "request" in p or "question" in p:
                    args[p] = "please solve the task"
                else:
                    args[p] = first if props[p].get("type") == "string" else 0
            msg["tool_calls"] = [{"id": "call_" + uuid.uuid4().hex[:8], "type": "function",
                                  "function": {"name": fn, "arguments": json.dumps(args)}}]
            finish = "tool_calls"
        elif _schema(req):
            # structured-output request (e.g. MAF's AgentOrchestrationOutput): the schema usually
            # forbids extra keys, so answer with exactly its properties and nothing else.
            sch = _schema(req)
            msg["content"] = json.dumps(_fill(sch, first, sch)); finish = "stop"
        else:
            body = {"next_speaker": first, "assignee_id": first, "choice": 1, "index": 0,
                    "selection": first, "reason": "mock", "is_request_satisfied": False,
                    "is_progress_being_made": True, "is_in_loop": False,
                    "instruction_or_question": "solve it", "agent": first}
            # a roster prompt that enumerates "Task ID: x" lines also gets a per-task assignment list
            body["assignments"] = [{"task_id": t, "assignee_id": first, "dependencies": []}
                                   for t in dict.fromkeys(TASK_ID.findall(json.dumps(req.get("messages", []))))]
            msg["content"] = json.dumps(body) + f"\n{first}"
            finish = "stop"
        if req.get("stream"):        # frameworks that stream agent runs (MAF handoff) need SSE, not JSON
            self._send_sse(msg, finish, req)
        else:
            self._send({"id": "chatcmpl-" + uuid.uuid4().hex[:8], "object": "chat.completion", "created": int(time.time()),
                        "model": req.get("model", "mock"), "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})

    def _send_sse(self, msg, finish, req):
        cid, model = "chatcmpl-" + uuid.uuid4().hex[:8], req.get("model", "mock")
        def chunk(delta, fin=None, usage=None):
            d = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
                 "choices": [] if usage else [{"index": 0, "delta": delta, "finish_reason": fin}]}
            if usage: d["usage"] = usage
            return f"data: {json.dumps(d)}\n\n".encode()
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        out = [chunk({"role": "assistant"})]
        if msg.get("content"): out.append(chunk({"content": msg["content"]}))
        out += [chunk({"tool_calls": [dict(tc, index=i)]}) for i, tc in enumerate(msg.get("tool_calls", []))]
        out.append(chunk({}, finish))
        if (req.get("stream_options") or {}).get("include_usage"):
            out.append(chunk(None, None, {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}))
        self.wfile.write(b"".join(out) + b"data: [DONE]\n\n")

    def _send(self, body):
        b = json.dumps(body).encode(); self.send_response(200)
        self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8123
    print(f"mock openai server on http://127.0.0.1:{port}/v1", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()   # threaded: frameworks fire concurrent calls
