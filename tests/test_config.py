"""Config parsing smoke tests: yaml loading, field validation, no hardcoded
constants leaking in (unknown keys must be rejected, not silently ignored)."""

from __future__ import annotations

import multiprocessing

import pytest
import yaml

from xgap_code.config import DemoReplayConfig


def _write_yaml(tmp_path, overrides=None):
    base = {
        "experiment_name": "test_exp",
        "local_output_root": str(tmp_path / "local"),
        "task_suites": ["libero_10"],
        "task_ids": {"libero_10": [0]},
        "episodes_per_task": 2,
        "control_modes": ["relative", "absolute"],
        "control_freq": 10,
        "init_states": True,
        "n_decision_points": 1,
        "exec_horizon": 1,
        "selection_unit": "chunk",
        "parallel_envs": "auto",
        "parallel_envs_cap": 4,
        "sync_every_n_episodes": 5,
        "checkpoint_name": "N/A_demo_replay",
        "checkpoint_hash": "N/A_demo_replay",
        "git_commit": "unknown",
        "random_seed": 0,
        "resume": True,
    }
    if overrides:
        base.update(overrides)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    return str(path)


def test_load_valid_config(tmp_path):
    path = _write_yaml(tmp_path)
    cfg = DemoReplayConfig.from_yaml(path)
    assert cfg.task_suites == ["libero_10"]
    assert cfg.control_modes == ["relative", "absolute"]
    assert cfg.n_decision_points == 1
    assert cfg.exec_horizon == 1
    assert cfg.selection_unit == "chunk"


def test_parallel_envs_auto_resolves_and_is_capped(tmp_path):
    path = _write_yaml(tmp_path, {"parallel_envs": "auto", "parallel_envs_cap": 2})
    cfg = DemoReplayConfig.from_yaml(path)
    assert cfg.parallel_envs <= 2
    assert cfg.parallel_envs <= multiprocessing.cpu_count()


def test_unknown_field_rejected(tmp_path):
    path = _write_yaml(tmp_path, {"totally_made_up_field": 123})
    with pytest.raises(ValueError, match="Unknown config fields"):
        DemoReplayConfig.from_yaml(path)


def test_invalid_control_mode_rejected(tmp_path):
    path = _write_yaml(tmp_path, {"control_modes": ["sideways"]})
    with pytest.raises(ValueError, match="invalid control_mode"):
        DemoReplayConfig.from_yaml(path)


def test_init_states_false_rejected(tmp_path):
    path = _write_yaml(tmp_path, {"init_states": False})
    with pytest.raises(ValueError, match="init_states=False"):
        DemoReplayConfig.from_yaml(path)


def test_invalid_selection_unit_rejected(tmp_path):
    path = _write_yaml(tmp_path, {"selection_unit": "not_a_real_unit"})
    with pytest.raises(ValueError, match="invalid selection_unit"):
        DemoReplayConfig.from_yaml(path)
