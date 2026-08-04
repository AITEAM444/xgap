#!/usr/bin/env python
"""Test 2 -- state save/restore determinism: does "same state + same action =
same result" actually hold for xgap's own env construction? This underlies
every future Oracle/World-model/Random candidate comparison (see README
"Design constraints", n_decision_points/exec_horizon/selection_unit) -- if
restoring a saved state and replaying the same action doesn't reproduce the
same outcome, comparing candidates branched from a shared decision point is
meaningless. Flagged in the project's own planning as the hardest
implementation element, and untouched until now.

Uses harness.get_sim_state/restore_sim_state (env.sim.get_state().flatten()
/ set_state_from_flattened(), the standard robosuite/MuJoCo idiom -- see
that function's own docstring for the "unverified against this specific
installed version" caveat; this script is the first real test of it).

Procedure, per trial:
  1. Reset env (standard init_state order, trial index as seed), step
     N_WARMUP_STEPS arbitrary actions to get away from the exact reset
     state (a fresh reset might trivially "match" even with a real bug
     elsewhere).
  2. Save state.
  3. Branch A: step ONE fixed test action, record resulting state/image.
  4. Restore the saved state.
  5. Branch B: step the SAME fixed test action again, record resulting
     state/image.
  6. Compare A vs B -- state (5-dim eef+gripper) and rendered image.

    python scripts/test_state_restore_determinism.py --task-suite libero_spatial --task-id 0 --n-trials 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from xgap_code.harness import get_sim_state, make_real_libero_env, restore_sim_state  # noqa: E402
from xgap_code.state_determinism import compare_images, compare_state_chunks  # noqa: E402

_N_WARMUP_STEPS = 10
# Fixed, arbitrary but non-trivial action (moves + closes the gripper) -- deliberately
# not all-zero, since a no-op action could trivially "match" without exercising physics.
_TEST_ACTION = np.array([0.1, 0.0, -0.1, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _extract_state(obs: dict) -> list[float]:
    """Same layout as replay.py's _extract_state -- not imported directly to
    avoid coupling this diagnostic script to demo-replay's module."""
    robot_state = obs["robot_state"]
    eef_pos = np.asarray(robot_state["eef"]["pos"], dtype=np.float32)
    gripper_qpos = np.asarray(robot_state["gripper"]["qpos"], dtype=np.float32)
    return np.concatenate([eef_pos, gripper_qpos]).tolist()


def run_one_trial(env, seed: int) -> dict:
    obs, _info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(_N_WARMUP_STEPS):
        warmup_action = rng.uniform(-0.2, 0.2, size=7).astype(np.float32)
        obs, _reward, terminated, truncated, _info = env.step(warmup_action)
        if terminated or truncated:
            return {"seed": seed, "skipped": "episode ended during warmup"}

    saved_state = get_sim_state(env)
    if saved_state is None:
        raise RuntimeError("get_sim_state returned None -- a real LiberoEnv is required, not MockLiberoEnv")

    obs_a, reward_a, _terminated_a, _truncated_a, _info_a = env.step(_TEST_ACTION)
    state_a = _extract_state(obs_a)
    image_a = env.render()

    restore_sim_state(env, saved_state)

    obs_b, reward_b, _terminated_b, _truncated_b, _info_b = env.step(_TEST_ACTION)
    state_b = _extract_state(obs_b)
    image_b = env.render()

    return {
        "seed": seed,
        "state_comparison": compare_state_chunks(state_a, state_b),
        "image_comparison": compare_images(image_a, image_b),
        "reward_identical": bool(np.isclose(reward_a, reward_b)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--control-mode", default="relative")
    parser.add_argument("--control-freq", type=int, default=20)
    args = parser.parse_args()

    results = []
    for trial in range(args.n_trials):
        env = make_real_libero_env(
            task_suite_name=args.task_suite,
            task_id=args.task_id,
            demo_episode_index_within_task=trial,
            control_mode=args.control_mode,
            control_freq=args.control_freq,
        )
        result = run_one_trial(env, seed=trial)
        env.close()
        results.append(result)
        print(result)

    valid = [r for r in results if "state_comparison" in r]
    n_state_identical = sum(1 for r in valid if r["state_comparison"]["identical"])
    n_image_identical = sum(1 for r in valid if r["image_comparison"]["identical"])
    print(
        f"\n[determinism] state identical: {n_state_identical}/{len(valid)}, "
        f"image identical: {n_image_identical}/{len(valid)} "
        f"({len(results) - len(valid)} trial(s) skipped: episode ended during warmup)"
    )


if __name__ == "__main__":
    main()
