"""Tests for the control_mode plumbing-verification check, the gripper
close-run duration metric, and the demo-vs-policy plot utility -- all added
per the follow-up instruction to distinguish "the flag never reached the env"
from "the flag reached the env and the physics genuinely tied/lost"."""

from __future__ import annotations

import numpy as np

from xgap_code.gripper_metrics import DEMO_MIN_ACTUATION_LAG_STEPS, longest_close_run
from xgap_code.harness import MockLiberoEnv, read_actual_control_mode
from xgap_code.plots import plot_episode_trajectory, plot_gripper_action_histogram
from xgap_code.replay import replay_episode, verify_control_mode_wiring


def test_read_actual_control_mode_reflects_correct_wiring():
    env = MockLiberoEnv(control_mode="absolute")
    env.reset(seed=0)
    assert read_actual_control_mode(env) == "absolute"


def test_read_actual_control_mode_exposes_wiring_bug():
    env = MockLiberoEnv(control_mode="absolute", simulate_wiring_bug=True)
    env.reset(seed=0)
    # Bug: reports "relative" regardless of what was requested.
    assert read_actual_control_mode(env) == "relative"


def test_verify_control_mode_wiring_passes_when_modes_differ():
    result = verify_control_mode_wiring({"relative": "relative", "absolute": "absolute"})
    assert result is None


def test_verify_control_mode_wiring_aborts_when_modes_collide():
    result = verify_control_mode_wiring({"relative": "relative", "absolute": "relative"})
    assert result is not None
    assert result["decision"] == "control_mode_not_wired"


def test_verify_control_mode_wiring_aborts_when_reading_unavailable():
    result = verify_control_mode_wiring({"relative": "relative", "absolute": None})
    assert result is not None
    assert result["decision"] == "control_mode_not_wired"


def test_longest_close_run_basic():
    actions = np.zeros((30, 7), dtype=np.float32)
    actions[:, 6] = -1.0
    actions[10:22, 6] = 1.0  # 12-step close run
    actions[25:27, 6] = 1.0  # 2-step close run
    assert longest_close_run(actions) == 12


def test_longest_close_run_never_closes():
    actions = np.zeros((10, 7), dtype=np.float32)
    actions[:, 6] = -1.0
    assert longest_close_run(actions) == 0


def test_longest_close_run_below_actuation_lag_is_detectable():
    """A close run shorter than the demo-measured actuation lag cannot have
    physically grasped -- this test just documents that the comparison is
    possible with plain numbers, not a claim about any real rollout."""
    actions = np.zeros((20, 7), dtype=np.float32)
    actions[:, 6] = -1.0
    actions[5:9, 6] = 1.0  # 4-step close run -- too short
    assert longest_close_run(actions) < DEMO_MIN_ACTUATION_LAG_STEPS


def test_plot_gripper_action_histogram_writes_file(tmp_path):
    demo = np.array([-1.0] * 20 + [1.0] * 20)
    policy = np.random.default_rng(0).normal(0, 0.1, size=40)  # clustered near 0 -- the bug case
    out_path = tmp_path / "gripper_hist.png"
    plot_gripper_action_histogram(demo, policy, str(out_path))
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_episode_trajectory_writes_file(tmp_path):
    state = [[0.0, 0.0, 0.0, 0.04, -0.04], [0.01, 0.0, 0.0, 0.0, 0.0]]
    action = [[0.0] * 7, [0.0] * 6 + [1.0]]
    out_path = tmp_path / "traj.png"
    plot_episode_trajectory(state, action, str(out_path), title="test")
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_episode_trajectory_with_comparison_series(tmp_path):
    state = [[0.0, 0.0, 0.0, 0.04, -0.04]] * 3
    action = [[0.0] * 7] * 3
    out_path = tmp_path / "traj_cmp.png"
    plot_episode_trajectory(
        state, action, str(out_path),
        compare_state_chunk=state, compare_action_chunk=action, compare_label="demo",
    )
    assert out_path.exists()


def test_replay_episode_saves_state_chunk_and_video(tmp_path):
    env = MockLiberoEnv(max_steps=10, require_final_gripper_close=True)
    actions = np.zeros((10, 7), dtype=np.float32)
    actions[:, 6] = -1.0
    actions[-1, 6] = 1.0
    video_path = tmp_path / "videos" / "ep0.mp4"

    record = replay_episode(
        env, actions,
        task_id="libero_10:0", task_suite="libero_10", episode_seed=0, environment_seed=0,
        condition="demo_replay_relative", control_mode="relative", control_freq=10,
        checkpoint_name="N/A_demo_replay", checkpoint_hash="N/A_demo_replay",
        git_commit="deadbeef", config_file="configs/demo_replay.yaml",
        video_path=video_path, video_sample_every_n_steps=3,
    )

    assert len(record.state_chunk) == record.rollout_length == 10
    assert len(record.state_chunk[0]) == 5  # eef_pos(3) + gripper_qpos(2)
    # steps 0, 3, 6, 9 sampled (every 3rd, 0-indexed)
    assert record.video_frame_steps == [0, 3, 6, 9]
    assert video_path.exists()
    assert video_path.stat().st_size > 0


def test_replay_episode_no_video_when_disabled(tmp_path):
    env = MockLiberoEnv(max_steps=5, require_final_gripper_close=False)
    actions = np.zeros((5, 7), dtype=np.float32)
    actions[:, 6] = -1.0

    record = replay_episode(
        env, actions,
        task_id="libero_10:0", task_suite="libero_10", episode_seed=0, environment_seed=0,
        condition="demo_replay_relative", control_mode="relative", control_freq=10,
        checkpoint_name="N/A_demo_replay", checkpoint_hash="N/A_demo_replay",
        git_commit="deadbeef", config_file="configs/demo_replay.yaml",
    )

    assert record.video_frame_steps == []
    assert len(record.state_chunk) == 5
