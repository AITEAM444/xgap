"""End-to-end smoke tests for the demo replay driver against MockLiberoEnv --
no mujoco/libero/lerobot installed. Validates: full episode loop via
replay_episode, the control_mode decision branching (winner vs. both-low), and
that run_demo_replay.run() resumes correctly across two invocations."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from xgap_code.harness import MockLiberoEnv
from xgap_code.replay import decide_env_convention, replay_episode, summarize_by_control_mode

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
import run_demo_replay  # noqa: E402
from xgap_code.config import DemoReplayConfig  # noqa: E402


def test_replay_episode_against_mock_env_success():
    env = MockLiberoEnv(max_steps=10, require_final_gripper_close=True)
    actions = np.zeros((10, 7), dtype=np.float32)
    actions[:, 6] = -1.0
    actions[-1, 6] = 1.0  # close on the final step -> mock env reports success

    record = replay_episode(
        env,
        actions,
        task_id="libero_10:0",
        task_suite="libero_10",
        episode_seed=0,
        environment_seed=0,
        condition="demo_replay_relative",
        control_mode="relative",
        control_freq=10,
        checkpoint_name="N/A_demo_replay",
        checkpoint_hash="N/A_demo_replay",
        git_commit="deadbeef",
        config_file="configs/demo_replay.yaml",
    )
    assert record.episode_success is True
    assert record.rollout_length == 10
    assert record.time_physics_s > 0
    assert record.time_reset_s > 0


def test_replay_episode_against_mock_env_failure():
    env = MockLiberoEnv(max_steps=10, require_final_gripper_close=True)
    actions = np.zeros((10, 7), dtype=np.float32)
    actions[:, 6] = -1.0  # never closes -> mock env reports failure

    record = replay_episode(
        env,
        actions,
        task_id="libero_10:0",
        task_suite="libero_10",
        episode_seed=1,
        environment_seed=1,
        condition="demo_replay_relative",
        control_mode="relative",
        control_freq=10,
        checkpoint_name="N/A_demo_replay",
        checkpoint_hash="N/A_demo_replay",
        git_commit="deadbeef",
        config_file="configs/demo_replay.yaml",
    )
    assert record.episode_success is False


def test_decide_env_convention_picks_higher_success_mode():
    rows = (
        [{"control_mode": "relative", "episode_success": True}] * 8
        + [{"control_mode": "relative", "episode_success": False}] * 2
        + [{"control_mode": "absolute", "episode_success": False}] * 10
    )
    summary = summarize_by_control_mode(rows)
    decision = decide_env_convention(summary)
    assert decision["decision"] == "control_mode_confirmed"
    assert decision["confirmed_control_mode"] == "relative"


def test_decide_env_convention_flags_exact_tie_as_inconclusive():
    rows = [{"control_mode": "relative", "episode_success": True}] * 10 + [
        {"control_mode": "absolute", "episode_success": True}
    ] * 10
    summary = summarize_by_control_mode(rows)
    decision = decide_env_convention(summary)
    assert decision["decision"] == "tie_inconclusive"


def test_decide_env_convention_flags_both_low_as_init_state_suspect():
    rows = [{"control_mode": "relative", "episode_success": False}] * 10 + [
        {"control_mode": "absolute", "episode_success": False}
    ] * 10
    summary = summarize_by_control_mode(rows)
    decision = decide_env_convention(summary)
    assert decision["decision"] == "both_low_suspect_init_state"
    assert "init" in decision["reason"].lower()


def _minimal_config_yaml(tmp_path) -> str:
    import yaml

    cfg = {
        "experiment_name": "mock_test",
        "local_output_root": str(tmp_path / "local"),
        "remote_output_root": str(tmp_path / "remote"),
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
        "parallel_envs_cap": 2,
        "sync_every_n_episodes": 1,
        "checkpoint_name": "N/A_demo_replay",
        "checkpoint_hash": "N/A_demo_replay",
        "git_commit": "unknown",
        "random_seed": 0,
        "resume": True,
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(path)


def test_full_mock_run_and_resume(tmp_path):
    config_path = _minimal_config_yaml(tmp_path)
    cfg = DemoReplayConfig.from_yaml(config_path)

    decision = run_demo_replay.run(cfg, config_file=config_path, mock=True)
    # MockLiberoEnv's success condition does not depend on control_mode at all, so a tie
    # between relative/absolute is the expected (and correctly detected) outcome here.
    assert decision["decision"] in (
        "control_mode_confirmed",
        "both_low_suspect_init_state",
        "tie_inconclusive",
    )

    n_files_first_run = len(list((tmp_path / "remote" / "episodes").glob("*.parquet")))
    expected = 1 * 2 * 2  # tasks(1) * episodes_per_task(2) * control_modes(2)
    assert n_files_first_run == expected

    # Second invocation with the same config must resume (skip all, write nothing new).
    cfg2 = DemoReplayConfig.from_yaml(config_path)
    run_demo_replay.run(cfg2, config_file=config_path, mock=True)
    n_files_second_run = len(list((tmp_path / "remote" / "episodes").glob("*.parquet")))
    assert n_files_second_run == n_files_first_run


def test_run_aborts_immediately_on_wiring_bug(tmp_path, monkeypatch):
    """If control_mode never reaches the env, run() must abort right after the
    first episode of each mode -- before running the rest of the sweep and
    before computing any success rate."""
    import xgap_code.harness as harness_module

    def _broken_mock_env(*, control_mode, **kwargs):
        return harness_module.MockLiberoEnv(control_mode=control_mode, simulate_wiring_bug=True)

    monkeypatch.setattr(run_demo_replay, "MockLiberoEnv", _broken_mock_env)

    config_path = _minimal_config_yaml(tmp_path)
    cfg = DemoReplayConfig.from_yaml(config_path)
    decision = run_demo_replay.run(cfg, config_file=config_path, mock=True)

    assert decision["decision"] == "control_mode_not_wired"
    # Only the 2 probe episodes (one per control_mode) should have run -- not the full
    # 1*2*2=4 episode sweep this config would otherwise produce.
    n_files = len(list((tmp_path / "remote" / "episodes").glob("*.parquet")))
    assert n_files == 2
