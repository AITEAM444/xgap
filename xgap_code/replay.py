"""Demo replay: step a real or mock LIBERO env through a recorded action
sequence, instrumented with per-stage timing, producing an EpisodeRecord.

This module only depends on the env-like reset()/step() contract -- it works
identically against harness.MockLiberoEnv (no simulator) and the real LiberoEnv
returned by harness.make_real_libero_env.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .logging_schema import EpisodeRecord
from .metering import StageTimer


def replay_episode(
    env: Any,
    demo_actions: np.ndarray,
    *,
    task_id: str,
    task_suite: str,
    episode_seed: int,
    environment_seed: int,
    condition: str,
    control_mode: str,
    control_freq: int,
    checkpoint_name: str,
    checkpoint_hash: str,
    git_commit: str,
    config_file: str,
) -> EpisodeRecord:
    timer = StageTimer()
    t_episode_start = time.perf_counter()

    with timer.stage("reset"):
        env.reset(seed=environment_seed)

    success = False
    executed_actions: list[list[float]] = []
    n_steps = 0
    for action in demo_actions:
        action = np.asarray(action, dtype=np.float32)
        with timer.stage("physics"):
            _obs, _reward, terminated, truncated, info = env.step(action)
        executed_actions.append(action.tolist())
        n_steps += 1
        if info.get("is_success", False):
            success = True
        if terminated or truncated:
            break

    execution_time = time.perf_counter() - t_episode_start

    return EpisodeRecord(
        task_id=task_id,
        task_suite=task_suite,
        episode_seed=episode_seed,
        environment_seed=environment_seed,
        condition=condition,
        action_chunk=executed_actions,
        episode_success=success,
        rollout_length=n_steps,
        checkpoint_name=checkpoint_name,
        checkpoint_hash=checkpoint_hash,
        git_commit=git_commit,
        config_file=config_file,
        execution_time=execution_time,
        control_mode=control_mode,
        control_freq=control_freq,
        **timer.as_fields(),
    )


def summarize_by_control_mode(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Success rate per control_mode, over whatever rows are given (tolerates
    a partial/interrupted run -- caller decides whether the sample is large
    enough to act on)."""
    by_mode: dict[str, list[bool]] = {}
    for row in rows:
        by_mode.setdefault(row["control_mode"], []).append(bool(row["episode_success"]))
    return {
        mode: {
            "n_episodes": len(successes),
            "success_rate": (sum(successes) / len(successes)) if successes else float("nan"),
        }
        for mode, successes in by_mode.items()
    }


# Below what success rate is a control_mode considered "not obviously working" --
# used only to decide the branch in decide_env_convention, not as a pass/fail
# threshold for anything else. Kept as a module constant (not config) because it
# encodes a judgment call about this specific diagnostic decision, not an
# experiment parameter.
_LOW_SUCCESS_THRESHOLD = 0.5


def decide_env_convention(mode_summary: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Given per-control_mode success rates from demo replay, decide which
    convention the environment/demo data agree on, per the user's explicit
    instruction: dual-run relative/absolute, adopt whichever is higher as the
    confirmed convention for downstream steps; if BOTH are low, the suspect is
    NOT control_mode but init-state setup, and this must be flagged rather
    than silently picking a "winner"."""
    if not mode_summary:
        return {"decision": "inconclusive", "reason": "no episodes recorded"}

    ranked = sorted(mode_summary.items(), key=lambda kv: kv[1]["success_rate"], reverse=True)
    best_mode, best_stats = ranked[0]
    all_low = all(stats["success_rate"] < _LOW_SUCCESS_THRESHOLD for _, stats in ranked)

    if len(ranked) > 1 and ranked[0][1]["success_rate"] == ranked[1][1]["success_rate"] and not all_low:
        # Exact tie: picking a "winner" here would be an arbitrary artifact of dict/list
        # ordering, not a real signal -- observed in practice with small mock samples.
        tied_modes = [m for m, stats in ranked if stats["success_rate"] == best_stats["success_rate"]]
        return {
            "decision": "tie_inconclusive",
            "reason": (
                f"Modes {tied_modes} tied at success_rate={best_stats['success_rate']:.1%}. Cannot "
                "confirm a convention from a tie -- run more episodes per task before proceeding to step 3."
            ),
            "mode_summary": mode_summary,
        }

    if all_low:
        return {
            "decision": "both_low_suspect_init_state",
            "reason": (
                "Both control_mode conditions show low success rate "
                f"(< {_LOW_SUCCESS_THRESHOLD:.0%}); per-instruction, this points at task "
                "init-state selection (or the demo/episode_index mapping -- see "
                "xgap_code/dataset_io.py docstring) rather than control_mode. Do not pick a "
                "winner; investigate init-state wiring before proceeding to step 3."
            ),
            "mode_summary": mode_summary,
        }

    return {
        "decision": "control_mode_confirmed",
        "confirmed_control_mode": best_mode,
        "reason": (
            f"'{best_mode}' had the higher demo-replay success rate "
            f"({best_stats['success_rate']:.1%} over {best_stats['n_episodes']} episodes); "
            "adopted as the environment/demo convention for subsequent steps. This is a "
            "harness-fidelity decision, not a policy-sweep result -- H4 (policy behavior "
            "under both control modes) is still evaluated separately in step 4."
        ),
        "mode_summary": mode_summary,
    }
