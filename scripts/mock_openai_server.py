"""Mock OpenAI-compatible chat server for testing framework workers WITHOUT a GPU.
Policy: if the request has `tools`, return a tool_call to the first tool whose name contains a
candidate-like token (transfer_to_*, handoff_to_*, agent_*), else the first tool; if no tools,
return a text/JSON answer naming the first `agent_XXXXXX` mentioned in the prompt (also as
{"next_speaker": ..., "assignee_id": ..., "choice": 0}). Usage: python scripts/mock_openai_server.py 8123
"""
import json, re, sys, time, uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

AGENT = re.compile(r"agent_\d{6}")


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        body = {"object": "list", "data": [{"id": "Qwen/Qwen2.5-7B-Instruct", "object": "model"}]} if "models" in self.path else {"ok": True}
        self._send(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); req = json.loads(self.rfile.read(n) or b"{}")
        text = json.dumps(req.get("messages", []))
        names = AGENT.findall(text)
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
        else:
            msg["content"] = json.dumps({"next_speaker": first, "assignee_id": first, "choice": 0, "index": 0,
                                         "selection": first, "reason": "mock", "is_request_satisfied": False,
                                         "is_progress_being_made": True, "is_in_loop": False,
                                         "instruction_or_question": "solve it"}) + f"\n{first}"
            finish = "stop"
        self._send({"id": "chatcmpl-" + uuid.uuid4().hex[:8], "object": "chat.completion", "created": int(time.time()),
                    "model": req.get("model", "mock"), "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})

    def _send(self, body):
        b = json.dumps(body).encode(); self.send_response(200)
        self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8123
    print(f"mock openai server on http://127.0.0.1:{port}/v1", flush=True)
    HTTPServer(("127.0.0.1", port), H).serve_forever()
