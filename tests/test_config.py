"""Config parsing smoke tests: yaml loading, field validation, no hardcoded
constants leaking in (unknown keys must be rejected, not silently ignored)."""

from __future__ import annotations

import multiprocessing

import pytest
import yaml

from xgap_code.config import (
    DemoReplayConfig,
    GateTwoDiversityConfig,
    GateTwoOutcomeConfig,
    InitStateSweepConfig,
    PolicyRolloutConfig,
)


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


def _write_sweep_yaml(tmp_path, overrides=None):
    base = {
        "experiment_name": "test_sweep",
        "local_output_root": str(tmp_path / "local"),
        "task_suite": "libero_10",
        "task_id": 0,
        "demo_within_task_index": 0,
        "control_mode": "relative",
        "control_freq": 20,
        "sync_every_n_episodes": 5,
        "checkpoint_name": "N/A_init_state_sweep",
        "checkpoint_hash": "N/A_init_state_sweep",
        "git_commit": "unknown",
        "resume": True,
    }
    if overrides:
        base.update(overrides)
    path = tmp_path / "sweep_cfg.yaml"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    return str(path)


def test_init_state_sweep_config_loads(tmp_path):
    path = _write_sweep_yaml(tmp_path)
    cfg = InitStateSweepConfig.from_yaml(path)
    assert cfg.task_suite == "libero_10"
    assert cfg.task_id == 0
    assert cfg.control_freq == 20


def test_init_state_sweep_config_rejects_unknown_field(tmp_path):
    path = _write_sweep_yaml(tmp_path, {"bogus_field": 1})
    with pytest.raises(ValueError, match="Unknown config fields"):
        InitStateSweepConfig.from_yaml(path)


def test_init_state_sweep_config_rejects_bad_control_mode(tmp_path):
    path = _write_sweep_yaml(tmp_path, {"control_mode": "diagonal"})
    with pytest.raises(ValueError, match="invalid control_mode"):
        InitStateSweepConfig.from_yaml(path)


def _write_rollout_yaml(tmp_path, overrides=None):
    base = {
        "experiment_name": "test_rollout",
        "local_output_root": str(tmp_path / "local"),
        "task_suite": "libero_spatial",
        "task_ids": [0],
        "checkpoint_path": "HuggingFaceVLA/smolvla_libero",
        "episodes_per_task": 5,
        "max_steps": 520,
        "control_mode": "relative",
        "control_freq": 20,
        "sync_every_n_episodes": 5,
        "resume": True,
    }
    if overrides:
        base.update(overrides)
    path = tmp_path / "rollout_cfg.yaml"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    return str(path)


def test_policy_rollout_config_loads(tmp_path):
    path = _write_rollout_yaml(tmp_path)
    cfg = PolicyRolloutConfig.from_yaml(path)
    assert cfg.task_suite == "libero_spatial"
    assert cfg.task_ids == [0]
    assert cfg.checkpoint_path == "HuggingFaceVLA/smolvla_libero"


def test_policy_rollout_config_rejects_unknown_field(tmp_path):
    path = _write_rollout_yaml(tmp_path, {"bogus_field": 1})
    with pytest.raises(ValueError, match="Unknown config fields"):
        PolicyRolloutConfig.from_yaml(path)


def test_policy_rollout_config_rejects_empty_task_ids(tmp_path):
    path = _write_rollout_yaml(tmp_path, {"task_ids": []})
    with pytest.raises(ValueError, match="task_ids must be non-empty"):
        PolicyRolloutConfig.from_yaml(path)


def test_policy_rollout_config_rejects_bad_control_mode(tmp_path):
    path = _write_rollout_yaml(tmp_path, {"control_mode": "sideways"})
    with pytest.raises(ValueError, match="invalid control_mode"):
        PolicyRolloutConfig.from_yaml(path)


def _write_gate2_yaml(tmp_path, overrides=None):
    base = {
        "experiment_name": "test_gate2",
        "local_output_root": str(tmp_path / "local"),
        "task_suite": "libero_spatial",
        "task_ids": [0, 2],
        "checkpoint_path": "HuggingFaceVLA/smolvla_libero",
        "source_output_root": str(tmp_path / "source"),
        "source_condition": "policy_rollout",
        "source_episode_seed": 0,
        "branch_step_fractions": [0.5],
        "n_candidates": 64,
        "exec_horizon": 10,
        "control_mode": "relative",
        "control_freq": 20,
        "gripper_dim": 6,
        "resume": True,
    }
    if overrides:
        base.update(overrides)
    path = tmp_path / "gate2_cfg.yaml"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    return str(path)


def test_gate2_diversity_config_loads(tmp_path):
    path = _write_gate2_yaml(tmp_path)
    cfg = GateTwoDiversityConfig.from_yaml(path)
    assert cfg.task_ids == [0, 2]
    assert cfg.n_candidates == 64


def test_gate2_diversity_config_rejects_unknown_field(tmp_path):
    path = _write_gate2_yaml(tmp_path, {"bogus_field": 1})
    with pytest.raises(ValueError, match="Unknown config fields"):
        GateTwoDiversityConfig.from_yaml(path)


def test_gate2_diversity_config_rejects_missing_source(tmp_path):
    path = _write_gate2_yaml(tmp_path, {"source_output_root": ""})
    with pytest.raises(ValueError, match="source_output_root"):
        GateTwoDiversityConfig.from_yaml(path)


def test_gate2_diversity_config_rejects_bad_branch_fraction(tmp_path):
    path = _write_gate2_yaml(tmp_path, {"branch_step_fractions": [1.5]})
    with pytest.raises(ValueError, match="branch_step_fractions"):
        GateTwoDiversityConfig.from_yaml(path)


def test_gate2_diversity_config_rejects_too_few_candidates(tmp_path):
    path = _write_gate2_yaml(tmp_path, {"n_candidates": 1})
    with pytest.raises(ValueError, match="n_candidates"):
        GateTwoDiversityConfig.from_yaml(path)


def _write_gate2_outcome_yaml(tmp_path, overrides=None):
    base = {
        "experiment_name": "test_gate2_outcome",
        "local_output_root": str(tmp_path / "local"),
        "task_suite": "libero_spatial",
        "task_ids": [0, 2],
        "checkpoint_path": "HuggingFaceVLA/smolvla_libero",
        "source_output_root": str(tmp_path / "source"),
        "source_condition": "policy_rollout",
        "source_episode_seed": 0,
        "branch_fraction": 0.5,
        "exec_horizon": 10,
        "max_outcome_trials": 5,
        "max_steps": 200,
        "control_mode": "relative",
        "control_freq": 20,
        "resume": True,
    }
    if overrides:
        base.update(overrides)
    path = tmp_path / "gate2_outcome_cfg.yaml"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    return str(path)


def test_gate2_outcome_config_loads(tmp_path):
    path = _write_gate2_outcome_yaml(tmp_path)
    cfg = GateTwoOutcomeConfig.from_yaml(path)
    assert cfg.task_ids == [0, 2]
    assert cfg.branch_fraction == 0.5
    assert cfg.max_outcome_trials == 5
    assert cfg.max_steps == 200


def test_gate2_outcome_config_rejects_unknown_field(tmp_path):
    path = _write_gate2_outcome_yaml(tmp_path, {"bogus_field": 1})
    with pytest.raises(ValueError, match="Unknown config fields"):
        GateTwoOutcomeConfig.from_yaml(path)


def test_gate2_outcome_config_rejects_missing_source(tmp_path):
    path = _write_gate2_outcome_yaml(tmp_path, {"source_output_root": ""})
    with pytest.raises(ValueError, match="source_output_root"):
        GateTwoOutcomeConfig.from_yaml(path)


def test_gate2_outcome_config_rejects_bad_branch_fraction(tmp_path):
    path = _write_gate2_outcome_yaml(tmp_path, {"branch_fraction": 1.5})
    with pytest.raises(ValueError, match="branch_fraction"):
        GateTwoOutcomeConfig.from_yaml(path)


def test_gate2_outcome_config_rejects_zero_max_outcome_trials(tmp_path):
    path = _write_gate2_outcome_yaml(tmp_path, {"max_outcome_trials": 0})
    with pytest.raises(ValueError, match="max_outcome_trials"):
        GateTwoOutcomeConfig.from_yaml(path)


def test_gate2_outcome_config_rejects_bad_control_mode(tmp_path):
    path = _write_gate2_outcome_yaml(tmp_path, {"control_mode": "sideways"})
    with pytest.raises(ValueError, match="invalid control_mode"):
        GateTwoOutcomeConfig.from_yaml(path)
