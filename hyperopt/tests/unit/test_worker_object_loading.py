import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("pymongo")

from hyperopt import mongoexp
from hyperopt.base import JOB_STATE_DONE, JOB_STATE_RUNNING, STATUS_OK


@pytest.mark.parametrize(
    "protocol", ["token_load", "bandit_json evaluate", "driver_attachment"]
)
def test_worker_loads_and_evaluates_dotted_object(protocol, tmp_path, monkeypatch):
    module_name = "hyperopt_dotted_worker_test"
    (tmp_path / (module_name + ".py")).write_text(
        "def objective(spec, ctrl):\n"
        "    return {'loss': spec['x'] ** 2, 'status': 'ok'}\n"
        "class Objective:\n"
        "    def __init__(self, offset=0.0):\n"
        "        self.offset = offset\n"
        "    def evaluate(self, spec, ctrl):\n"
        "        return {'loss': spec['x'] ** 2 + self.offset, 'status': 'ok'}\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    name = module_name + (".objective" if protocol == "token_load" else ".Objective")
    command = (protocol, "objective" if protocol == "driver_attachment" else name)
    job = {
        "_id": "objective-job",
        "state": JOB_STATE_RUNNING,
        "exp_key": "test",
        "misc": {"cmd": command, "vals": {"x": [3.0]}},
    }
    attachments = {"objective": mongoexp.pickler.dumps((name, (), {"offset": 2.0}))}

    class Jobs:
        def reserve(self, host_id, exp_key):
            return job

        def refresh(self, current):
            assert current is job

        def update(self, current, values):
            assert current is job
            current.update(values)

    jobs = Jobs()
    monkeypatch.setattr(
        mongoexp,
        "MongoTrials",
        lambda handle, **kwargs: SimpleNamespace(
            handle=handle, attachments=attachments
        ),
    )
    worker = mongoexp.MongoWorker(jobs, workdir=str(tmp_path), logfilename=None)
    try:
        worker.run_one()
        assert job["state"] == JOB_STATE_DONE
        expected_loss = 11.0 if protocol == "driver_attachment" else 9.0
        assert job["result"] == {"loss": expected_loss, "status": STATUS_OK}
    finally:
        sys.modules.pop(module_name, None)
