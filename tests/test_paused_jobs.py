import json

import pytest

from vig_dagger.config import DEFAULTS
from vig_dagger.jobs import Job, Queue


def test_pause_survives_queue_restart_without_releasing_dependencies(tmp_path, monkeypatch):
    jobs = [Job('train', ['worker'], kind='train'), Job('eval', ['eval'], deps=['train'])]
    queue = Queue(jobs, {**DEFAULTS['resources'], 'paused_jobs': ['train']}, tmp_path)
    monkeypatch.setattr(queue, '_adopt', lambda: None)
    monkeypatch.setattr(queue, '_start', lambda *args: pytest.fail('Paused job or dependent started'))
    class EndPoll(Exception):
        pass
    def stop_poll(_):
        raise EndPoll
    monkeypatch.setattr('vig_dagger.jobs.time.sleep', stop_poll)
    with pytest.raises(EndPoll):
        queue.run()
    state = json.loads((tmp_path / 'queue_state.json').read_text())['jobs']
    assert state['train']['state'] == 'paused'
    assert state['eval']['state'] == 'waiting'
    assert not queue.done and not queue.failed
    resumed = Queue(jobs, DEFAULTS['resources'], tmp_path)
    resumed.write_state()
    assert json.loads((tmp_path / 'queue_state.json').read_text())['jobs']['train']['state'] == 'ready'


def test_pauses_outside_selected_subgraph_do_not_block_it(tmp_path):
    queue = Queue([Job('selected', ['worker'])],
                  {**DEFAULTS['resources'], 'paused_jobs': ['not_selected']}, tmp_path)
    assert not queue.paused
    queue.write_state()
    assert json.loads((tmp_path / 'queue_state.json').read_text())['jobs']['selected']['state'] == 'ready'
