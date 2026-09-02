"""CAMEL Workforce recipe: the coordinator reads `<node_id>:<description>:<toolkits>` and answers
`assignee_id`, so the candidate name goes in node_id. The callback fires on assignment, before the
task is posted to the channel, so raising there stops the run with nothing executed."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _bridge import serve_worker      # noqa: E402
from _wk import openai_kwargs         # noqa: E402


class Picked(Exception):
    pass


def _callback():
    """WorkforceCallback is an ABC with 11 abstract log_* hooks; only one of them does anything."""
    from camel.societies.workforce.workforce_callback import WorkforceCallback
    cls = type("PickCallback", (WorkforceCallback,),
               {**{m: (lambda self, *a, **kw: None) for m in dir(WorkforceCallback) if m.startswith("log_")},
                "log_task_assigned": lambda self, e: (_ for _ in ()).throw(Picked(e.worker_id))})
    return cls()


def select(req):
    from camel.agents import ChatAgent
    from camel.models import ModelFactory
    from camel.societies.workforce import Workforce
    from camel.tasks import Task
    from camel.types import ModelPlatformType
    kw = openai_kwargs(req)
    model = ModelFactory.create(model_platform=ModelPlatformType.VLLM, model_type=kw["model"],
                                url=kw["base_url"], api_key=kw["api_key"],
                                model_config_dict={"temperature": 0.0})
    wf = Workforce("router", coordinator_agent=ChatAgent(model=model), task_agent=ChatAgent(model=model),
                   default_model=model, callbacks=[_callback()])
    for c in req["candidates"]:
        wf.add_single_agent_worker(c["description"], worker=ChatAgent(model=model))
        wf._children[-1].node_id = c["name"]   # the only place the coordinator can read a candidate name
    try:
        wf.process_task(Task(content=req["task"], id="t0"))
    except Picked as p:
        return p.args[0], f"TaskAssignedEvent.worker_id={p.args[0]}"
    finally:
        wf.stop()


if __name__ == "__main__":
    from camel.logger import disable_logging
    disable_logging()
    serve_worker(select)
