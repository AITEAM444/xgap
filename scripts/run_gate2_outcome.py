#!/usr/bin/env python
"""Gate 2, outcome-based verdict: does candidate diversity actually change
whether the episode succeeds, not just whether the action values differ?

GateTwoDiversityConfig's (scripts/run_gate2_diversity.py) endpoint-variance
metric is INVALIDATED for verdict purposes -- confirmed from real data:
`frac_commanding_close` (gripper_channel_distribution) went 0.53 at
branch_fraction=0.2 to 0.85 at 0.5, meaning candidates really do differ on
WHEN they command the gripper closed. But `exec_horizon=10` is shorter than
the gripper's own actuation lag (15-20 steps, see
gripper_metrics.DEMO_MIN_ACTUATION_LAG_STEPS) -- that timing difference has
no window to show up as a physical endpoint difference. Only episode
success/failure decides Gate 2 now.

**Two design bugs from the first version of this script, both fixed:**
1. Only 5 trials per point -- too few to conclude "no diversity" from an
   all-same result. Now `n_candidates` (default 64), still early-stopping
   the moment a mix IS observed (that's already conclusive; the fix only
   raises the ceiling for the all-same case).
2. Source episodes were always a single SUCCESS episode -- branching at
   50% into an already-successful trajectory near-guarantees the handed-
   off base policy also succeeds, regardless of which candidate ran. That
   tests "does resuming a good trajectory still work", not "does candidate
   choice matter" -- unrelated to Gate 2. `source_episode_seeds` is now a
   dict of task_id -> list of seeds mixing real recorded SUCCESS and
   FAILURE episodes (see `scripts/list_source_episode_outcomes.py` to find
   real ones instead of guessing). Failure-source points are the ones that
   matter: "was about to fail, but a good candidate flipped it to success"
   is the literal mechanism an Oracle curve measures.

**A third bug, caught after task 0's real failure seeds turned out to all
have `rollout_length=520` (ran the full Gate-1 cap):** `max_steps` was
being applied to the base-policy handoff as an ABSOLUTE cap from episode
step 0, not a budget from the branch point. Harmless for a short
SUCCESS-source prefix, but at `branch_fraction=0.5` a 520-step failure
source gives a 260-step prefix alone -- already past a 200 absolute cap
before the handoff runs one step, so every candidate returned
`success=False` regardless of what it did. Fixed in
`run_one_candidate_outcome`: the handoff's `max_steps` is now
`step_count + max_steps` (budget counted from the branch point), matching
what `GateTwoOutcomeConfig.max_steps`'s own comment always said.

Procedure, per (task_id, source_episode_seed):
  1. Reach the branch point via prefix replay (same real recorded Gate-1
     episode, same construction as run_gate2_diversity.py) -- fresh env,
     Test 2's binding design rule.
  2. Sample ONE candidate action chunk (predict_action_chunk(noise=None) --
     fresh flow-matching noise every call, confirmed from source).
  3. Commit the candidate's first `exec_horizon` steps.
  4. Hand off to the BASE POLICY (closed-loop,
     policy_rollout.step_with_policy_until_done) for the rest of the
     episode -- literally "Oracle" in miniature.
  5. Record episode_success. Repeat with a FRESH candidate up to
     `n_candidates`, stopping EARLY once both a success and a failure have
     been observed among the outcomes so far.

Verdict:
  - `mixed` = outcomes contain both a success and a failure.
  - `is_primary_signal` = `mixed AND the source episode was a FAILURE` --
    this is what actually decides Gate 2 (a mixed result branching from an
    already-successful source is recorded but not decisive on its own).
  - Gate 2 PASSES if `is_primary_signal` is true for at least one
    (task, seed) point.
  - All candidates at a point landing the SAME result is evidence against
    usefulness at that specific point, regardless of the source's own
    outcome -- see README "Gate 2, corrected" for the full reasoning.

    python scripts/run_gate2_outcome.py --config configs/gate2_outcome.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
) -> tuple[bool, int]:
    """Fresh env, reach branch state, sample ONE candidate chunk, commit its
    first exec_horizon steps, then hand off to the base policy for up to
    `max_steps` MORE steps (a budget counted from the branch point, not
    from episode step 0 -- see bug note below) until success/termination.
    Returns (episode_success, total_steps_run) -- total_steps_run includes
    the prefix, so it's directly comparable to the source episode's own
    rollout_length.

    BUG FIXED HERE, not a hypothetical: `max_steps` was originally passed
    straight through to step_with_policy_until_done as an ABSOLUTE cap
    (start_step=step_count, max_steps=max_steps) -- fine for a short
    SUCCESS-source prefix (~40-50 steps), but a FAILURE source can have
    rollout_length=520 (ran the full Gate-1 cap): at branch_fraction=0.5
    that alone is a 260-step prefix, already past a 200 absolute cap
    before the handoff runs a single step -- every candidate would return
    `success=False` regardless of what it actually does, making the
    failure-source test (the one that decides Gate 2) measure nothing.
    Fixed by giving the handoff its own `max_steps`-sized budget added ON
    TOP of wherever the branch point landed."""
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
                return success, step_count

        # Clear the policy's own action queue before handing off to closed-loop
        # control -- candidate sampling used predict_action_chunk() directly and
        # never touched it, but stale queue state from a PREVIOUS trial's handoff
        # could otherwise leak into this one.
        policy.reset()
        handoff_success, n_steps, _actions, _states = step_with_policy_until_done(
            env,
            obs,
            policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            env_preprocessor=env_preprocessor,
            preprocess_observation_fn=preprocess_observation_fn,
            task_description=task_description,
            max_steps=step_count + max_steps,
            start_step=step_count,
        )
        return success or handoff_success, n_steps
    finally:
        env.close()


def run_one_branch_point(
    *,
    task_id: int,
    task_description: str,
    seed: int,
    cfg: GateTwoOutcomeConfig,
    policy,
    preprocessor,
    postprocessor,
    env_preprocessor,
    preprocess_observation_fn,
    out_dir: Path,
) -> dict:
    """One (task_id, source_episode_seed) branch point: up to cfg.n_candidates
    fresh-candidate trials, early-stopping once mixed. Returns the point's
    result dict (also written to out_dir)."""
    source_episode = load_source_episode(cfg.source_output_root, cfg.source_condition, cfg.task_suite, task_id, seed)
    source_success = bool(source_episode["episode_success"])
    full_action_chunk = source_episode["action_chunk"]
    rollout_length = source_episode["rollout_length"]
    prefix_length = max(1, min(rollout_length - 1, round(cfg.branch_fraction * rollout_length)))
    prefix_actions = full_action_chunk[:prefix_length]

    result_path = out_dir / f"task{task_id}_seed{seed}_frac{cfg.branch_fraction}_outcomes.json"
    if cfg.resume and result_path.exists():
        print(f"  seed={seed}: resumed from {result_path}")
        return json.loads(result_path.read_text())

    print(f"  seed={seed} source_episode_success={source_success} rollout_length={rollout_length} "
          f"(prefix_length={prefix_length})")

    outcomes: list[bool] = []
    for trial in range(cfg.n_candidates):
        t_start = time.perf_counter()
        success, total_steps = run_one_candidate_outcome(
            task_suite=cfg.task_suite,
            task_id=task_id,
            episode_seed=seed,
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
        elapsed = time.perf_counter() - t_start
        outcomes.append(success)
        print(f"    candidate={trial + 1}/{cfg.n_candidates} success={success} "
              f"rollout_length={total_steps} elapsed={elapsed:.1f}s")
        if any(outcomes) and not all(outcomes):
            print(f"  seed={seed}: mixed outcome reached after {trial + 1} candidate(s) -- stopping early")
            break

    mixed = any(outcomes) and not all(outcomes)
    is_primary_signal = mixed and not source_success
    point = {
        "task_id": task_id,
        "source_episode_seed": seed,
        "source_episode_success": source_success,
        "branch_fraction": cfg.branch_fraction,
        "prefix_length": prefix_length,
        "n_candidates_run": len(outcomes),
        "outcomes": outcomes,
        "n_success": sum(outcomes),
        "mixed": mixed,
        "is_primary_signal": is_primary_signal,
    }
    result_path.write_text(json.dumps(point, indent=2))
    print(f"  seed={seed}: {point['n_success']}/{len(outcomes)} candidate successes, "
          f"mixed={mixed}, is_primary_signal={is_primary_signal}")
    return point


def run(cfg: GateTwoOutcomeConfig) -> dict:
    out_dir = Path(cfg.local_output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_results = []
    for task_id in cfg.task_ids:
        task_description = get_libero_task_language(cfg.task_suite, task_id)
        seeds = cfg.source_episode_seeds[task_id]
        print(f"\n[gate2-outcome] task={task_id} '{task_description}', source_episode_seeds={seeds}")

        (
            policy, preprocessor, postprocessor, env_preprocessor, _env_postprocessor, preprocess_observation_fn,
        ) = build_policy_and_processors(
            checkpoint_path=cfg.checkpoint_path,
            task_suite_name=cfg.task_suite,
            task_id=task_id,
            control_freq=cfg.control_freq,
        )

        points = [
            run_one_branch_point(
                task_id=task_id,
                task_description=task_description,
                seed=seed,
                cfg=cfg,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                env_preprocessor=env_preprocessor,
                preprocess_observation_fn=preprocess_observation_fn,
                out_dir=out_dir,
            )
            for seed in seeds
        ]

        task_pass = any(p["is_primary_signal"] for p in points)
        task_result = {"task_id": task_id, "task_description": task_description, "points": points, "task_pass": task_pass}
        print(f"  [gate2-outcome] task={task_id}: task_pass={task_pass} "
              f"({sum(p['is_primary_signal'] for p in points)}/{len(points)} points are primary-signal passes)")
        task_results.append(task_result)

    gate2_pass = any(t["task_pass"] for t in task_results)
    summary = {"tasks": task_results, "gate2_pass": gate2_pass}
    summary_path = out_dir / "gate2_outcome_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[gate2-outcome] gate2_pass={gate2_pass} "
          f"({sum(t['task_pass'] for t in task_results)}/{len(task_results)} tasks show a primary-signal pass)")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = GateTwoOutcomeConfig.from_yaml(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
