#!/usr/bin/env python
"""Test 2 (revised): prefix-replay determinism -- NOT state save/restore.

DESIGN CHANGE (see README "Test 2: design change to prefix replay"): state
save/restore (harness.get_sim_state/restore_sim_state --
env.sim.get_state()/set_state_from_flattened() + sim.forward()) was tested
and left real, non-decaying residual noise (signal/noise ratio as low as
1.0x at n_compare_steps=10) even after restoring qacc_warmstart AND the
robot controller's own array/scalar state (goal_pos, goal_ori, joint_pos,
mass_matrix, etc. -- confirmed captured, not missing). The interpolator
hypothesis was checked and ruled out too (interpolator_pos/interpolator_ori
are both None -- not in use by this controller). The remaining candidate
is MuJoCo/robosuite API-level: sim.forward() may not fully reconstruct
whatever internal state a normal step() sequence leaves behind. Root-
causing that would take days with no guaranteed fix, four weeks out from
the gate -- not worth it when there's a construction that sidesteps the
problem entirely.

Sidestep: instead of restoring a saved mid-episode state, each candidate
replays from env.reset(seed) through a FIXED, recorded prefix action
sequence up to the branch point, then diverges with its own candidate
action(s). Same seed + same action sequence -> MuJoCo is deterministic BY
CONSTRUCTION, so there is no restore step and no hidden internal state to
miss -- the state at the branch point is exactly what a normal step()
sequence produces, because it IS one. Test 2's actual requirement (fair
candidate comparison from a shared decision point) is satisfied by this
construction, not by achieving bit-identical state restoration.

This script verifies that determinism claim directly, not just assumes
it: run the SAME (seed, prefix action sequence, post-branch action
sequence) through env.reset()+step() TWICE and check the two runs are
BIT-IDENTICAL -- exactly 0 residual at every compared step, not "small".
Anything nonzero means something in this project's own harness
construction (not the MuJoCo/mujoco_py internals implicated by Test 2's
original state-restore result) is introducing real nondeterminism, and
needs fixing before candidate comparison can be trusted.

    python scripts/test_prefix_replay_determinism.py --task-suite libero_spatial --task-id 0 --n-trials 3 --prefix-steps 10 --post-branch-steps 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from xgap_code.harness import make_real_libero_env  # noqa: E402
from xgap_code.state_determinism import compare_state_chunks  # noqa: E402


def _extract_state(obs: dict) -> list[float]:
    """Same layout as replay.py's _extract_state -- not imported directly to
    avoid coupling this diagnostic script to demo-replay's module."""
    robot_state = obs["robot_state"]
    eef_pos = np.asarray(robot_state["eef"]["pos"], dtype=np.float32)
    gripper_qpos = np.asarray(robot_state["gripper"]["qpos"], dtype=np.float32)
    return np.concatenate([eef_pos, gripper_qpos]).tolist()


def run_one_trial(
    *, task_suite: str, task_id: int, control_mode: str, control_freq: int,
    seed: int, prefix_steps: int, post_branch_steps: int,
) -> dict:
    prefix_rng = np.random.default_rng(seed)
    prefix_actions = prefix_rng.uniform(-0.2, 0.2, size=(prefix_steps, 7)).astype(np.float32)
    # Different RNG stream from the prefix's -- the "candidate" action(s) after the branch point.
    post_rng = np.random.default_rng(seed + 1000)
    post_actions = post_rng.uniform(-0.15, 0.15, size=(post_branch_steps, 7)).astype(np.float32)
    post_actions[:, 6] = 1.0  # keep commanding "close" throughout -- non-trivial gripper dim too
    full_sequence = np.concatenate([prefix_actions, post_actions], axis=0)

    def run_once() -> list[list[float]]:
        # HYPOTHESIS TEST (see README): reusing ONE env instance across two
        # reset(seed=X) calls (the previous version of this script) gave
        # different starting scenes despite the identical seed, and neither
        # passing seed nor seeding the numpy global RNG changed that. New
        # suspect: LiberoEnv may track an internal per-INSTANCE reset counter
        # that advances init_state selection on every reset() call regardless
        # of the seed argument (a common pattern for cycling through episodes
        # in continuous training loops) -- which a freshly CONSTRUCTED env
        # would not carry between the two runs. Testing directly: build a
        # brand-new env for every run_once() call instead of reusing one.
        env = make_real_libero_env(
            task_suite_name=task_suite,
            task_id=task_id,
            demo_episode_index_within_task=seed,
            control_mode=control_mode,
            control_freq=control_freq,
        )
        try:
            obs, _info = env.reset(seed=seed)
            # index 0 = the RESET observation itself, before any action -- isolates
            # "does reset(seed=X) alone already differ" from "the divergence only
            # starts once actions are applied".
            states = [_extract_state(obs)]
            for action in full_sequence:
                obs, _reward, terminated, truncated, _info = env.step(action)
                states.append(_extract_state(obs))
                if terminated or truncated:
                    break
            return states
        finally:
            env.close()

    run_1 = run_once()
    run_2 = run_once()

    n = min(len(run_1), len(run_2))
    diffs = [compare_state_chunks(run_1[i], run_2[i])["max_abs_diff"] for i in range(n)]
    return {"seed": seed, "n_steps": n, "diffs": diffs, "reset_diff": diffs[0] if diffs else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--n-trials", type=int, default=3)
    parser.add_argument("--prefix-steps", type=int, default=10)
    parser.add_argument("--post-branch-steps", type=int, default=10)
    parser.add_argument("--control-mode", default="relative")
    parser.add_argument("--control-freq", type=int, default=20)
    args = parser.parse_args()

    results = []
    for trial in range(args.n_trials):
        result = run_one_trial(
            task_suite=args.task_suite,
            task_id=args.task_id,
            control_mode=args.control_mode,
            control_freq=args.control_freq,
            seed=trial,
            prefix_steps=args.prefix_steps,
            post_branch_steps=args.post_branch_steps,
        )
        results.append(result)

        diffs = [round(d, 10) for d in result["diffs"]]
        all_exact_zero = all(d == 0.0 for d in result["diffs"])
        reset_exact_zero = result["reset_diff"] == 0.0
        print(f"seed={trial} n_steps={result['n_steps']} all_exact_zero={all_exact_zero}")
        print(f"  reset_diff (index 0, before any action)={round(result['reset_diff'], 10)} exact_zero={reset_exact_zero}")
        print(f"  diff_per_step (index 0 = reset, 1..N = post-action)={diffs}")

    all_zero = all(all(d == 0.0 for d in r["diffs"]) for r in results)
    max_diff = max((d for r in results for d in r["diffs"]), default=0.0)
    print(f"\n[prefix-replay] {len(results)} trials. all exactly zero: {all_zero}. max diff observed: {max_diff:.2e}")
    if all_zero:
        print("[prefix-replay] PASS -- reset()+replay is bit-identical. Prefix replay is a valid candidate-comparison construction.")
    else:
        print("[prefix-replay] FAIL -- nondeterminism exists even without any state restore. Root cause is in this harness, not MuJoCo's set_state/forward.")


if __name__ == "__main__":
    main()
