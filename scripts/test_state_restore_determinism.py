#!/usr/bin/env python
"""Test 2 -- state save/restore determinism, signal-vs-noise: is the
save/restore mechanism precise enough to trust candidate comparisons built
on top of it? This underlies every future Oracle/World-model/Random
candidate comparison (see README "Design constraints",
n_decision_points/exec_horizon/selection_unit) -- if restoring a saved
state and replaying an action doesn't reproduce the same outcome,
comparing candidates branched from a shared decision point is meaningless.
Flagged in the project's own planning as the hardest implementation
element.

EARLIER RESULT (1 step, same-chunk-only comparison): NOT bit-identical --
small state (~0.0015) and image (~0.2% pixels) differences, consistent
across trials. Reward matched in all cases, but that's NOT reassuring on
its own: LIBERO's success signal is binary/coarse (a few mm of object
displacement can't be detected by it), so reward-matching says nothing
about whether the underlying state noise is small enough to trust.

The only metric that actually matters: the noise from restore-imprecision
has to be small RELATIVE TO the signal from two genuinely different
candidates, not small in some absolute sense. This script now measures
both directly, from the SAME saved state:

  - SIGNAL: run two DIFFERENT fixed action chunks (candidate 1 vs
    candidate 2, as if two different policy/world-model proposals) for
    n_compare_steps each -- the resulting state divergence between them
    is what a real candidate comparison would be measuring.
  - NOISE: run candidate 1's chunk TWICE (once, then again after a
    restore) -- the resulting state divergence is restore-imprecision
    alone, the same action chosen both times.

  ratio = signal / noise, at every step. Per the project's own threshold:
  ratio ~100x is practically safe, ~10x is risky, ~1x means candidate
  comparison doesn't work at all -- this ratio (not either raw number) is
  Test 2's actual pass/fail criterion.

Uses harness.get_sim_state/restore_sim_state (env.sim.get_state().flatten()
/ set_state_from_flattened(), the standard robosuite/MuJoCo idiom -- see
that function's own docstring for the "unverified against this specific
installed version" caveat).

Procedure, per trial:
  1. Reset env (standard init_state order, trial index as seed), step
     N_WARMUP_STEPS arbitrary actions to get away from the exact reset
     state.
  2. Save state.
  3. Branch A: step chunk_1 (n_compare_steps), recording state/image at
     every step.
  4. Restore the saved state.
  5. Branch B: step chunk_2 (DIFFERENT from chunk_1), same recording.
  6. Restore the saved state again.
  7. Branch C: step chunk_1 AGAIN (same as branch A).
  8. signal = compare(A, B) per step; noise = compare(A, C) per step.

    python scripts/test_state_restore_determinism.py --task-suite libero_spatial --task-id 0 --n-trials 3 --n-compare-steps 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from xgap_code.harness import get_sim_state, make_real_libero_env, restore_sim_state  # noqa: E402
from xgap_code.state_determinism import compare_state_chunks  # noqa: E402

_N_WARMUP_STEPS = 10


def _extract_state(obs: dict) -> list[float]:
    """Same layout as replay.py's _extract_state -- not imported directly to
    avoid coupling this diagnostic script to demo-replay's module."""
    robot_state = obs["robot_state"]
    eef_pos = np.asarray(robot_state["eef"]["pos"], dtype=np.float32)
    gripper_qpos = np.asarray(robot_state["gripper"]["qpos"], dtype=np.float32)
    return np.concatenate([eef_pos, gripper_qpos]).tolist()


def _make_action_chunk(rng: np.random.Generator, n_steps: int) -> np.ndarray:
    chunk = rng.uniform(-0.15, 0.15, size=(n_steps, 7)).astype(np.float32)
    chunk[:, 6] = 1.0  # keep commanding "close" throughout -- non-trivial gripper dim too
    return chunk


def _run_branch(env, actions: np.ndarray) -> list[list[float]]:
    states = []
    for action in actions:
        obs, _reward, terminated, truncated, _info = env.step(action)
        states.append(_extract_state(obs))
        if terminated or truncated:
            break
    return states


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
    qacc = saved_state.get("qacc_warmstart")
    if qacc is None:
        print(f"  seed={seed}: qacc_warmstart NOT FOUND on sim.data -- fix is a no-op for this trial")
    else:
        print(f"  seed={seed}: qacc_warmstart captured, shape={qacc.shape}, nonzero={np.count_nonzero(qacc)}/{qacc.size}")

    # Two genuinely different fixed chunks -- different RNG streams, both distinct from warmup's.
    chunk_1 = _make_action_chunk(np.random.default_rng(seed + 1000), n_compare_steps)
    chunk_2 = _make_action_chunk(np.random.default_rng(seed + 2000), n_compare_steps)

    branch_a = _run_branch(env, chunk_1)  # candidate 1

    restore_sim_state(env, saved_state)
    branch_b = _run_branch(env, chunk_2)  # candidate 2 -- diverges from A by real action difference (SIGNAL)

    restore_sim_state(env, saved_state)
    branch_c = _run_branch(env, chunk_1[: len(branch_a)])  # candidate 1 again -- diverges from A by restore noise only (NOISE)

    n = min(len(branch_a), len(branch_b), len(branch_c))
    per_step = []
    for i in range(n):
        signal = compare_state_chunks(branch_a[i], branch_b[i])["max_abs_diff"]
        noise = compare_state_chunks(branch_a[i], branch_c[i])["max_abs_diff"]
        ratio = signal / max(noise, 1e-12)
        per_step.append({"step": i, "signal": signal, "noise": noise, "ratio": ratio})

    return {"seed": seed, "n_steps_compared": n, "per_step": per_step}


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
        signals = [round(s["signal"], 6) for s in result["per_step"]]
        noises = [round(s["noise"], 6) for s in result["per_step"]]
        ratios = [round(s["ratio"], 1) for s in result["per_step"]]
        print(f"seed={trial} n_steps={result['n_steps_compared']}")
        print(f"  signal_per_step={signals}")
        print(f"  noise_per_step={noises}")
        print(f"  signal_noise_ratio_per_step={ratios}")

    valid = [r for r in results if "per_step" in r]
    if valid:
        all_ratios = [s["ratio"] for r in valid for s in r["per_step"]]
        last_step_ratios = [r["per_step"][-1]["ratio"] for r in valid]
        min_ratio = min(all_ratios)
        print(
            f"\n[determinism] {len(valid)}/{len(results)} trials valid. "
            f"min signal/noise ratio (any step, any trial): {min_ratio:.1f}x, "
            f"avg ratio at last compared step: {np.mean(last_step_ratios):.1f}x"
        )
        if min_ratio >= 100:
            print("[determinism] >= 100x -- practically safe for candidate comparison.")
        elif min_ratio >= 10:
            print("[determinism] 10-100x -- risky; candidate comparisons near this ratio may be unreliable.")
        else:
            print("[determinism] < 10x -- candidate comparison is not meaningful at this noise level.")


if __name__ == "__main__":
    main()
