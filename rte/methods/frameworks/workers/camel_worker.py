"""CAMEL Workforce rival (SPEC §6A recipe 9), run inside $RTE_DATA/env/fw_camel.

A `Workforce` over the top-k candidates as `SingleAgentWorker` children. Its coordinator prompt
(`ASSIGN_TASK_PROMPT`) renders the roster as `<node_id>:<description>:<toolkits>` and must answer with
JSON `assignments[].assignee_id`, so the candidate name goes in `node_id` and the self-description in
the worker's `description`. A `WorkforceCallback` raises out of `log_task_assigned`, which the workforce
fires immediately after assignment and before posting the task to the channel: no worker ever runs.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _bridge import serve_worker   # noqa: E402

_MODEL = {}


class Picked(Exception):
    def __init__(self, worker_id):
        self.worker_id = worker_id


def _callback_cls():
    from camel.societies.workforce.workforce_callback import WorkforceCallback
    noops = {m: (lambda self, *a, **kw: None) for m in dir(WorkforceCallback) if m.startswith("log_")}
    return type("PickCallback", (WorkforceCallback,),
                {**noops, "log_task_assigned": lambda self, event: (_ for _ in ()).throw(Picked(event.worker_id))})


def _model(req):
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType
    key = (req["model"], req["base_url"], req["api_key"])
    if _MODEL.get("key") != key:
        _MODEL["key"] = key
        _MODEL["m"] = ModelFactory.create(model_platform=ModelPlatformType.VLLM, model_type=req["model"],
                                          url=req["base_url"], api_key=req["api_key"],
                                          model_config_dict={"temperature": 0.0})
        _MODEL["cb"] = _callback_cls()
    return _MODEL["m"]


def select(req):
    from camel.agents import ChatAgent
    from camel.societies.workforce import Workforce
    from camel.tasks import Task
    model = _model(req)
    wf = Workforce("router", coordinator_agent=ChatAgent(model=model), task_agent=ChatAgent(model=model),
                   default_model=model, callbacks=[_MODEL["cb"]()])
    for c in req["candidates"]:
        wf.add_single_agent_worker(c["description"], worker=ChatAgent(model=model))
        wf._children[-1].node_id = c["name"]        # only place the coordinator can read a candidate name
    try:
        wf.process_task(Task(content=req["task"], id="t0"))
    except Picked as p:
        return p.worker_id, f"TaskAssignedEvent.worker_id={p.worker_id}"
    finally:
        wf.stop()
    return None


if __name__ == "__main__":
    from camel.logger import disable_logging
    disable_logging()
    serve_worker(select)
