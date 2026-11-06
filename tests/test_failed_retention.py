"""Failed trajectories remain available for analysis and cannot enter retraining."""
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from vig_dagger.isaac import Recorder
from vig_dagger.steps import dagger_sources


def test_training_selection_excludes_recorded_failures(tmp_path):
    episodes = tmp_path / 'episodes'
    episodes.mkdir()
    rows = [
        dict(seed=1, success=False, finished_at=1, file='trajectories/mixed.hdf5', traj=0),
        dict(seed=2, success=True, finished_at=3, file='trajectories/mixed.hdf5', traj=2),
        dict(seed=3, success=True, finished_at=2, file='trajectories/mixed.hdf5', traj=1),
        dict(seed=4, success=False, finished_at=4, file='trajectories/mixed.hdf5', traj=3),
    ]
    (episodes / 'worker_00.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    assert dagger_sources(tmp_path, 2) == [(tmp_path / 'trajectories/mixed.hdf5', [1, 2])]
    with pytest.raises(SystemExit, match='2 successful episodes'):
        dagger_sources(tmp_path, 3)


def test_real_isaac_hdf5_preserves_failed_payload_and_outcome(tmp_path):
    # Load the real dataset-only package without initializing unrelated USD/rendering imports.
    import importlib.util
    import sys
    source = Path(__file__).parents[1] / 'third_party/IsaacLab/source/isaaclab/isaaclab/utils/datasets'
    if not source.exists():
        pytest.skip('Isaac Lab source is not installed')
    spec = importlib.util.spec_from_file_location('_retention_isaac_datasets', source / '__init__.py',
                                                 submodule_search_locations=[str(source)])
    datasets = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = datasets
    spec.loader.exec_module(datasets)
    handler = datasets.HDF5DatasetFileHandler()
    handler.create(str(tmp_path / 'recordings.hdf5'))

    class Manager:
        _dataset_file_handler = handler
        episode = None

        def record_pre_reset(self, ids, force_export_or_skip=False):
            pass

        def set_success_to_episodes(self, ids, successes):
            self.episode.success = bool(successes[0, 0])

        def export_episodes(self, ids):
            handler.write_episode(self.episode)

    manager = Manager()
    recorder = Recorder(SimpleNamespace(recorder_manager=manager, device='cpu'))
    for expected_index, success in enumerate([False, True]):
        episode = datasets.EpisodeData()
        episode.data = {
            'actions': torch.ones(2, 7),
            'obs': {'state': torch.ones(2, 9), 'rgb_camera': {
                'left_cam': torch.ones(2, 4, 4, 3, dtype=torch.uint8),
                'wrist_cam': torch.ones(2, 4, 4, 3, dtype=torch.uint8)}},
            'states': {'joint_pos': torch.ones(2, 9)},
        }
        manager.episode = episode
        index = recorder.export({'scene_seed': expected_index, 'outcome': 'success' if success else 'time_limit_policy'},
                                {'dagger': {'controller': np.array([0, 1])}}, success=success)
        assert index == expected_index
        group = handler._hdf5_data_group[f'demo_{index}']
        assert bool(group.attrs['success']) is success
        assert group['actions'].shape == (2, 7)
        assert group['obs/rgb_camera/left_cam'].shape == (2, 4, 4, 3)
        assert group['obs/rgb_camera/wrist_cam'].shape == (2, 4, 4, 3)
        assert group['states/joint_pos'].shape == (2, 9)
        assert group['dagger/controller'][:].tolist() == [0, 1]
    recorder.close()
    with h5py.File(tmp_path / 'recordings.hdf5') as f:
        assert len(f['data']) == 2
        assert not bool(f['data/demo_0'].attrs['success'])
        assert bool(f['data/demo_1'].attrs['success'])
