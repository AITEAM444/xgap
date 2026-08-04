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
installed version" caveat).

FIRST RESULT (n_compare_steps=1, single step post-restore): NOT bit-identical
-- small but consistent state (~0.0015) and image (~0.2% pixels) differences
across all 5 trials, reward identical in all 5. Most likely explanation:
mujoco_py's MjSimState (time, qpos, qvel, act) does not include the contact
solver's warm-start acceleration (qacc_warmstart) -- physics is deterministic
given the TRUE internal solver state, but this flattened snapshot may not
capture all of it. THIS SCRIPT now extends the comparison across
`--n-compare-steps` (not just 1) to check whether that small gap compounds,
stays flat, or shrinks over an exec_horizon-scale rollout, which is what
actually matters for candidate comparison (not a single step).

Procedure, per trial:
  1. Reset env (standard init_state order, trial index as seed), step
     N_WARMUP_STEPS arbitrary actions to get away from the exact reset
     state (a fresh reset might trivially "match" even with a real bug
     elsewhere).
  2. Save state.
  3. Branch A: step a FIXED sequence of `--n-compare-steps` test actions,
     recording state/image/reward after EVERY step.
  4. Restore the saved state.
  5. Branch B: step the SAME fixed action sequence again, same recording.
  6. Compare A vs B at every step index -- watch whether max_abs_diff grows,
     stays flat, or shrinks across the sequence.

    python scripts/test_state_restore_determinism.py --task-suite libero_spatial --task-id 0 --n-trials 3 --n-compare-steps 10
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


def _extract_state(obs: dict) -> list[float]:
    """Same layout as replay.py's _extract_state -- not imported directly to
    avoid coupling this diagnostic script to demo-replay's module."""
    robot_state = obs["robot_state"]
    eef_pos = np.asarray(robot_state["eef"]["pos"], dtype=np.float32)
    gripper_qpos = np.asarray(robot_state["gripper"]["qpos"], dtype=np.float32)
    return np.concatenate([eef_pos, gripper_qpos]).tolist()


def _run_branch(env, actions: np.ndarray) -> list[dict]:
    steps = []
    for action in actions:
        obs, reward, terminated, truncated, _info = env.step(action)
        steps.append({"state": _extract_state(obs), "image": env.render(), "reward": reward})
        if terminated or truncated:
            break
    return steps


def run_one_trial(env, seed: int, n_compare_steps: int) -> dict:
    obs, _info = env.reset(seed=seed)
    warmup_rng = np.random.default_rng(seed)
    for _ in range(_N_WARMUP_STEPS):
        warmup_action = warmup_rng.uniform(-0.2, 0.2, size=7).astype(np.float32)
        obs, _reward, terminated, truncated, _info = env.step(warmup_action)
        if terminated or truncated:
            return {"seed": seed, "skipped": "episode ended during warmup"}

    saved_state = get_sim_state(env)
    if saved_state is None:
        raise RuntimeError("get_sim_state returned None -- a real LiberoEnv is required, not MockLiberoEnv")

    # Fixed, deterministic, non-trivial action sequence -- SAME sequence replayed on both
    # branches. A different RNG stream from warmup (seed+1000) so it doesn't just repeat it.
    compare_rng = np.random.default_rng(seed + 1000)
    test_actions = compare_rng.uniform(-0.15, 0.15, size=(n_compare_steps, 7)).astype(np.float32)
    test_actions[:, 6] = 1.0  # keep commanding "close" throughout -- non-trivial gripper dim too

    branch_a = _run_branch(env, test_actions)

    restore_sim_state(env, saved_state)

    branch_b = _run_branch(env, test_actions[: len(branch_a)])

    per_step = [
        {
            "step": i,
            "state_comparison": compare_state_chunks(a["state"], b["state"]),
            "image_comparison": compare_images(a["image"], b["image"]),
            "reward_identical": bool(np.isclose(a["reward"], b["reward"])),
        }
        for i, (a, b) in enumerate(zip(branch_a, branch_b, strict=True))
    ]

    return {"seed": seed, "n_steps_compared": len(per_step), "per_step": per_step}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--n-trials", type=int, default=3)
    parser.add_argument("--n-compare-steps", type=int, default=10)
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
        result = run_one_trial(env, seed=trial, n_compare_steps=args.n_compare_steps)
        env.close()
        results.append(result)

        if "per_step" not in result:
            print(result)
            continue
        state_diffs = [s["state_comparison"]["max_abs_diff"] for s in result["per_step"]]
        image_pct_diffs = [s["image_comparison"]["pct_pixels_differing"] for s in result["per_step"]]
        rewards_matched = [s["reward_identical"] for s in result["per_step"]]
        print(f"seed={trial} n_steps={result['n_steps_compared']}")
        print(f"  state_max_abs_diff_per_step={[round(d, 6) for d in state_diffs]}")
        print(f"  image_pct_differing_per_step={[round(d, 4) for d in image_pct_diffs]}")
        print(f"  reward_identical_per_step={rewards_matched}")

    valid = [r for r in results if "per_step" in r]
    if valid:
        first_step_diffs = [r["per_step"][0]["state_comparison"]["max_abs_diff"] for r in valid]
        last_step_diffs = [r["per_step"][-1]["state_comparison"]["max_abs_diff"] for r in valid]
        print(
            f"\n[determinism] {len(valid)}/{len(results)} trials valid. "
            f"avg state diff at step 0: {np.mean(first_step_diffs):.6f}, "
            f"avg state diff at last step: {np.mean(last_step_diffs):.6f} "
            f"(growth ratio: {np.mean(last_step_diffs) / max(np.mean(first_step_diffs), 1e-12):.2f}x)"
        )


if __name__ == "__main__":
    main()
