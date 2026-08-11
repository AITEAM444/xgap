#!/usr/bin/env python
"""Gate 2 (candidate diversity): does the policy actually propose
meaningfully different candidates from the same observation, or does it
collapse to a single mode? If candidates are indistinguishable, there is
nothing for an Oracle/World-model/Random comparison to select between, and
there is no reason to draw an Oracle curve -- see README "Gate 2:
candidate diversity".

Raised as a live concern: task 1's Gate-1 run showed 19/20 episodes with
near-identical rollout_length (95-121) and longest_close_run_steps
(39-63) -- a possible sign the policy converges to a single behavioral
mode. This checks whether that tendency shows up in the actual experiment-
set tasks too.

For each task, branch points are read from an ALREADY-RECORDED real
episode (a prior scripts/run_policy_rollout.py run's own EpisodeStore
output) -- not synthesized -- via prefix replay (see harness.py's
STANDARD-order docstring and README "Test 2: design change to prefix
replay" for why this reaches the exact real state deterministically): a
fresh env, reset, replay the recorded action_chunk up to a chosen step.

Diversity comes ONLY from the policy's own intrinsic stochasticity.
SmolVLAPolicy.predict_action_chunk(batch, noise=None) was confirmed from
source (not guessed) to sample fresh flow-matching noise internally on
every call and to bypass the action queue entirely (unlike
select_action()) -- calling it N times on the SAME observation gives N
independently-sampled candidate chunks. No noise-scale/temperature knobs
here on purpose -- inflating them would contaminate any later
multimodality analysis.

Four metrics per (task, branch point), computed over `--n-candidates`
sampled chunks:
  - per-dimension action std
  - mean pairwise L2 distance between candidate chunks
  - gripper channel (action[:,6]) distribution
  - trajectory ENDPOINT variance -- requires actually EXECUTING each
    candidate's first `--exec-horizon` steps from the branch point (fresh
    env per candidate, per Test 2's binding design rule -- reusing one env
    instance across candidates would reintroduce the exact nondeterminism
    Test 2 eliminated) and recording where the arm ends up.

DIAGNOSTIC ONLY -- does NOT decide Gate 2 pass/fail. Real results (task
0/2, branch_fraction=0.2) showed the endpoint_variance metric is
INVALIDATED for verdict purposes: `frac_commanding_close` went 0.53 (at
0.2) to 0.85 (at 0.5), i.e. candidates really do differ on WHEN they
command the gripper closed, but `exec_horizon=10` is shorter than the
gripper's own actuation lag (15-20 steps) -- that real timing difference
has no way to show up as a physical endpoint difference within 10 steps.
Endpoint-variance measurement stops once branch_step_fractions=0.7
finishes running; kept only as the evidence for why
scripts/run_gate2_outcome.py (success/fail, not action/endpoint metrics)
is the actual Gate 2 verdict now -- see that script's module docstring
and README "Gate 2: real results, and redesign to outcome-based verdict".

    python scripts/run_gate2_diversity.py --config configs/gate2_diversity.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from xgap_code.config import GateTwoDiversityConfig  # noqa: E402
from xgap_code.diversity_metrics import (  # noqa: E402
    endpoint_variance,
    gripper_channel_distribution,
    mean_pairwise_l2,
    per_dimension_std,
)
from xgap_code.harness import get_libero_task_language, make_real_libero_env  # noqa: E402
from xgap_code.logging_schema import episode_key  # noqa: E402
from xgap_code.policy_rollout import build_policy_and_processors, build_policy_observation  # noqa: E402


def _extract_state(obs: dict) -> list[float]:
    """Same layout as replay.py's/policy_rollout.py's _extract_state."""
    robot_state = obs["robot_state"]
    eef_pos = np.asarray(robot_state["eef"]["pos"], dtype=np.float32)
    gripper_qpos = np.asarray(robot_state["gripper"]["qpos"], dtype=np.float32)
    return np.concatenate([eef_pos, gripper_qpos]).tolist()


def load_source_episode(
    source_output_root: str, condition: str, task_suite: str, task_id: int, episode_seed: int
) -> dict:
    task_label = f"{task_suite}:{task_id}"
    key = episode_key(condition, task_suite, task_label, episode_seed)
    path = Path(source_output_root) / "episodes" / f"{key}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No recorded source episode at {path} -- run scripts/run_policy_rollout.py for "
            f"({task_suite}, task {task_id}, episode_seed {episode_seed}) first."
        )
    return pq.read_table(path).to_pylist()[0]


def reach_branch_state(
    *, task_suite: str, task_id: int, episode_seed: int, control_mode: str, control_freq: int, prefix_actions
):
    """Fresh env (Test 2's binding design rule), reset + replay prefix_actions.
    Returns (env, obs) with the env still OPEN at the branch point -- caller
    must env.close() when done with it."""
    env = make_real_libero_env(
        task_suite_name=task_suite,
        task_id=task_id,
        demo_episode_index_within_task=episode_seed,
        control_mode=control_mode,
        control_freq=control_freq,
    )
    obs, _info = env.reset(seed=episode_seed)
    for action in prefix_actions:
        obs, _reward, terminated, truncated, _info = env.step(np.asarray(action, dtype=np.float32))
        if terminated or truncated:
            env.close()
            raise RuntimeError(
                "Episode ended during prefix replay -- branch point unreachable, "
                "lower this branch_step_fraction."
            )
    return env, obs


def sample_candidates(
    *, policy, preprocessor, postprocessor, env_preprocessor, preprocess_observation_fn,
    obs: dict, task_description: str, n_candidates: int,
) -> np.ndarray:
    """Returns (n_candidates, T, action_dim) -- T is whatever
    predict_action_chunk returns (the full predicted chunk, not truncated
    to n_action_steps -- that truncation only happens inside select_action's
    queue-population step, confirmed from source)."""
    import torch

    batched_obs = build_policy_observation(
        obs,
        preprocess_observation_fn=preprocess_observation_fn,
        env_preprocessor=env_preprocessor,
        preprocessor=preprocessor,
        task_description=task_description,
    )
    chunks = []
    for _ in range(n_candidates):
        with torch.inference_mode():
            raw_chunk = policy.predict_action_chunk(batched_obs)  # (1, T, action_dim), pre-postprocessing
        # Postprocessor is only verified (elsewhere in this project) against (batch, action_dim)
        # tensors -- rollout_policy_episode always applies it to one already-dequeued action, never
        # a full (batch, T, action_dim) chunk. Loop per-step rather than risk an unverified
        # broadcasting assumption over the T axis.
        t_steps = raw_chunk.shape[1]
        processed_steps = [postprocessor(raw_chunk[:, t, :]) for t in range(t_steps)]
        processed_chunk = torch.stack(processed_steps, dim=1)  # (1, T, action_dim)
        chunks.append(processed_chunk[0].to("cpu").numpy())
    return np.stack(chunks, axis=0)


def measure_endpoints(
    *, task_suite: str, task_id: int, episode_seed: int, control_mode: str, control_freq: int,
    prefix_actions, candidate_chunks: np.ndarray, exec_horizon: int,
) -> np.ndarray:
    """Executes each candidate's first `exec_horizon` steps from a FRESH env
    reset to the branch point (one fresh env per candidate -- see module
    docstring). Returns (n_candidates, 5) -- _extract_state's [eef_xyz,
    gripper_qpos] layout, after execution."""
    endpoints = []
    for candidate in candidate_chunks:
        env, obs = reach_branch_state(
            task_suite=task_suite,
            task_id=task_id,
            episode_seed=episode_seed,
            control_mode=control_mode,
            control_freq=control_freq,
            prefix_actions=prefix_actions,
        )
        try:
            n_steps = min(exec_horizon, candidate.shape[0])
            for t in range(n_steps):
                obs, _reward, terminated, truncated, _info = env.step(candidate[t].astype(np.float32))
                if terminated or truncated:
                    break
            endpoints.append(_extract_state(obs))
        finally:
            env.close()
    return np.asarray(endpoints, dtype=np.float64)


def run(cfg: GateTwoDiversityConfig) -> dict:
    out_dir = Path(cfg.local_output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for task_id in cfg.task_ids:
        task_description = get_libero_task_language(cfg.task_suite, task_id)
        print(f"\n[gate2] task={task_id} '{task_description}'")

        source_episode = load_source_episode(
            cfg.source_output_root, cfg.source_condition, cfg.task_suite, task_id, cfg.source_episode_seed
        )
        full_action_chunk = source_episode["action_chunk"]
        rollout_length = source_episode["rollout_length"]
        print(f"  source episode: rollout_length={rollout_length}, success={source_episode['episode_success']}")

        (
            policy, preprocessor, postprocessor, env_preprocessor, _env_postprocessor, preprocess_observation_fn,
        ) = build_policy_and_processors(
            checkpoint_path=cfg.checkpoint_path,
            task_suite_name=cfg.task_suite,
            task_id=task_id,
            control_freq=cfg.control_freq,
        )

        for fraction in cfg.branch_step_fractions:
            prefix_length = max(1, min(rollout_length - 1, round(fraction * rollout_length)))
            prefix_actions = full_action_chunk[:prefix_length]
            result_key = f"task{task_id}_frac{fraction}"
            result_path = out_dir / f"{result_key}.json"
            if cfg.resume and result_path.exists():
                print(f"  branch fraction={fraction} (prefix_length={prefix_length}): resumed from {result_path}")
                all_results.append(json.loads(result_path.read_text()))
                continue

            print(f"  branch fraction={fraction} (prefix_length={prefix_length}): reaching branch state...")
            env, obs = reach_branch_state(
                task_suite=cfg.task_suite,
                task_id=task_id,
                episode_seed=cfg.source_episode_seed,
                control_mode=cfg.control_mode,
                control_freq=cfg.control_freq,
                prefix_actions=prefix_actions,
            )
            env.close()  # sampling only needs the observation, not a live env

            print(f"  sampling {cfg.n_candidates} candidates...")
            candidates = sample_candidates(
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                env_preprocessor=env_preprocessor,
                preprocess_observation_fn=preprocess_observation_fn,
                obs=obs,
                task_description=task_description,
                n_candidates=cfg.n_candidates,
            )

            print(f"  executing {cfg.n_candidates} candidates x {cfg.exec_horizon} steps for endpoints...")
            endpoints = measure_endpoints(
                task_suite=cfg.task_suite,
                task_id=task_id,
                episode_seed=cfg.source_episode_seed,
                control_mode=cfg.control_mode,
                control_freq=cfg.control_freq,
                prefix_actions=prefix_actions,
                candidate_chunks=candidates,
                exec_horizon=cfg.exec_horizon,
            )

            result = {
                "task_id": task_id,
                "task_description": task_description,
                "branch_fraction": fraction,
                "prefix_length": prefix_length,
                "n_candidates": cfg.n_candidates,
                "per_dimension_std": per_dimension_std(candidates),
                "mean_pairwise_l2": mean_pairwise_l2(candidates),
                "gripper_distribution": gripper_channel_distribution(candidates, gripper_dim=cfg.gripper_dim),
                "endpoint_variance": endpoint_variance(endpoints),
            }
            result_path.write_text(json.dumps(result, indent=2))
            print(f"  action_std={[round(s, 4) for s in result['per_dimension_std']]}")
            print(f"  mean_pairwise_l2={result['mean_pairwise_l2']:.4f}")
            print(f"  gripper={result['gripper_distribution']}")
            print(f"  endpoint_total_variance={result['endpoint_variance']['total_variance']:.6f}, "
                  f"endpoint_mean_pairwise_l2={result['endpoint_variance']['mean_pairwise_l2']:.4f}")
            all_results.append(result)

    summary_path = out_dir / "gate2_summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2))
    return {"results": all_results, "summary_path": str(summary_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = GateTwoDiversityConfig.from_yaml(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
