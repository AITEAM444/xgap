#!/usr/bin/env python
"""Gate 2, outcome-based verdict: does candidate diversity actually change
whether the episode succeeds, not just whether the action values differ?

GateTwoDiversityConfig's (scripts/run_gate2_diversity.py) endpoint-variance
metric is INVALIDATED for verdict purposes, not just ambiguous -- confirmed
from real data: `frac_commanding_close` (gripper_channel_distribution) went
0.53 at branch_fraction=0.2 to 0.85 at 0.5, meaning candidates really do
differ on WHEN they command the gripper closed. But `exec_horizon=10` is
shorter than the gripper's own actuation lag (15-20 steps, see
gripper_metrics.DEMO_MIN_ACTUATION_LAG_STEPS) -- a real timing difference
between candidates has no way to show up as a physical endpoint difference
within 10 steps. The metric was measuring the wrong window, not reporting a
true absence of consequential diversity. Endpoint-variance measurement
stops once branch_step_fractions=0.7 finishes; it's kept only as
diagnostic evidence for why this outcome-based script exists, not as a
Gate 2 input. Only episode success/failure decides Gate 2 now.

Procedure, per (task, branch_fraction):
  1. Reach the branch point via prefix replay (same real recorded Gate-1
     episode, same construction as run_gate2_diversity.py).
  2. Sample ONE candidate action chunk (predict_action_chunk(noise=None) --
     fresh flow-matching noise every call, confirmed from source).
  3. Commit the candidate's first `exec_horizon` steps.
  4. Hand off to the BASE POLICY (closed-loop,
     policy_rollout.step_with_policy_until_done) for the rest of the
     episode -- literally "Oracle" in miniature: one candidate branch,
     finished by the same policy that would run anyway.
  5. Record episode_success.
  6. Repeat with a FRESH candidate (fresh env per Test 2's binding design
     rule) up to `max_outcome_trials`, stopping EARLY as soon as both a
     success and a failure have been observed.

Verdict:
  - All trials at a (task, branch_fraction) point come back the SAME
    (all success or all failure) -> Gate 2 fails at that point --
    candidates that differ in action space converge to the same outcome
    anyway, nothing for Oracle/World-model/Random to select between.
  - A mix of success and failure -> Gate 2 passes -- and this result IS
    already the first real Oracle-curve data point (candidates exist whose
    outcomes differ, so "pick the best one" is a meaningful operation).

    python scripts/run_gate2_outcome.py --config configs/gate2_outcome.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from xgap_code.config import GateTwoOutcomeConfig  # noqa: E402
from xgap_code.harness import get_libero_task_language  # noqa: E402
from xgap_code.policy_rollout import (  # noqa: E402
    build_policy_and_processors,
    build_policy_observation,
    step_with_policy_until_done,
)
from run_gate2_diversity import load_source_episode, reach_branch_state  # noqa: E402


def run_one_candidate_outcome(
    *,
    task_suite: str,
    task_id: int,
    episode_seed: int,
    control_mode: str,
    control_freq: int,
    prefix_actions,
    policy,
    preprocessor,
    postprocessor,
    env_preprocessor,
    preprocess_observation_fn,
    task_description: str,
    exec_horizon: int,
    max_steps: int,
) -> bool:
    """Fresh env, reach branch state, sample ONE candidate chunk, commit its
    first exec_horizon steps, then hand off to the base policy until
    success/termination/max_steps. Returns episode_success."""
    import torch

    env, obs = reach_branch_state(
        task_suite=task_suite,
        task_id=task_id,
        episode_seed=episode_seed,
        control_mode=control_mode,
        control_freq=control_freq,
        prefix_actions=prefix_actions,
    )
    try:
        batched_obs = build_policy_observation(
            obs,
            preprocess_observation_fn=preprocess_observation_fn,
            env_preprocessor=env_preprocessor,
            preprocessor=preprocessor,
            task_description=task_description,
        )
        with torch.inference_mode():
            raw_chunk = policy.predict_action_chunk(batched_obs)  # (1, T, action_dim)
        # Per-step postprocess -- see run_gate2_diversity.sample_candidates for why
        # (postprocessor only verified elsewhere against (batch, action_dim) tensors).
        t_steps = raw_chunk.shape[1]
        processed_steps = [postprocessor(raw_chunk[:, t, :]) for t in range(t_steps)]
        candidate = torch.stack(processed_steps, dim=1)[0].to("cpu").numpy()  # (T, action_dim)

        success = False
        step_count = len(prefix_actions)
        n_committed = min(exec_horizon, candidate.shape[0])
        for t in range(n_committed):
            obs, _reward, terminated, truncated, info = env.step(candidate[t].astype(np.float32))
            step_count += 1
            if info.get("is_success", False):
                success = True
            if terminated or truncated:
                return success

        # Clear the policy's own action queue before handing off to closed-loop
        # control -- candidate sampling used predict_action_chunk() directly and
        # never touched it, but stale queue state from a PREVIOUS trial's handoff
        # could otherwise leak into this one.
        policy.reset()
        handoff_success, _n_steps, _actions, _states = step_with_policy_until_done(
            env,
            obs,
            policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            env_preprocessor=env_preprocessor,
            preprocess_observation_fn=preprocess_observation_fn,
            task_description=task_description,
            max_steps=max_steps,
            start_step=step_count,
        )
        return success or handoff_success
    finally:
        env.close()


def run(cfg: GateTwoOutcomeConfig) -> dict:
    out_dir = Path(cfg.local_output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_results = []
    for task_id in cfg.task_ids:
        task_description = get_libero_task_language(cfg.task_suite, task_id)
        print(f"\n[gate2-outcome] task={task_id} '{task_description}'")

        source_episode = load_source_episode(
            cfg.source_output_root, cfg.source_condition, cfg.task_suite, task_id, cfg.source_episode_seed
        )
        full_action_chunk = source_episode["action_chunk"]
        rollout_length = source_episode["rollout_length"]
        prefix_length = max(1, min(rollout_length - 1, round(cfg.branch_fraction * rollout_length)))
        prefix_actions = full_action_chunk[:prefix_length]
        print(f"  rollout_length={rollout_length}, branch_fraction={cfg.branch_fraction} "
              f"(prefix_length={prefix_length})")

        result_path = out_dir / f"task{task_id}_frac{cfg.branch_fraction}_outcomes.json"
        if cfg.resume and result_path.exists():
            print(f"  resumed from {result_path}")
            task_results.append(json.loads(result_path.read_text()))
            continue

        (
            policy, preprocessor, postprocessor, env_preprocessor, _env_postprocessor, preprocess_observation_fn,
        ) = build_policy_and_processors(
            checkpoint_path=cfg.checkpoint_path,
            task_suite_name=cfg.task_suite,
            task_id=task_id,
            control_freq=cfg.control_freq,
        )

        outcomes: list[bool] = []
        for trial in range(cfg.max_outcome_trials):
            success = run_one_candidate_outcome(
                task_suite=cfg.task_suite,
                task_id=task_id,
                episode_seed=cfg.source_episode_seed,
                control_mode=cfg.control_mode,
                control_freq=cfg.control_freq,
                prefix_actions=prefix_actions,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                env_preprocessor=env_preprocessor,
                preprocess_observation_fn=preprocess_observation_fn,
                task_description=task_description,
                exec_horizon=cfg.exec_horizon,
                max_steps=cfg.max_steps,
            )
            outcomes.append(success)
            print(f"  trial={trial} success={success}")
            if any(outcomes) and not all(outcomes):
                print(f"  mixed outcome reached after {trial + 1} trial(s) -- stopping early")
                break

        mixed = any(outcomes) and not all(outcomes)
        result = {
            "task_id": task_id,
            "task_description": task_description,
            "branch_fraction": cfg.branch_fraction,
            "prefix_length": prefix_length,
            "n_trials": len(outcomes),
            "outcomes": outcomes,
            "n_success": sum(outcomes),
            "mixed": mixed,
            "verdict": "PASS" if mixed else "FAIL",
        }
        result_path.write_text(json.dumps(result, indent=2))
        print(f"  [gate2-outcome] task={task_id}: {result['n_success']}/{len(outcomes)} succeeded -> {result['verdict']}")
        task_results.append(result)

    overall_pass = any(r["mixed"] for r in task_results)
    summary = {"tasks": task_results, "any_task_passed": overall_pass}
    summary_path = out_dir / "gate2_outcome_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[gate2-outcome] {sum(r['mixed'] for r in task_results)}/{len(task_results)} tasks show mixed outcomes.")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = GateTwoOutcomeConfig.from_yaml(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
