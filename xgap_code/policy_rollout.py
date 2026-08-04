"""Live policy rollout: step a real LIBERO env through actions actually
selected by a policy (not pre-recorded demo actions -- see replay.py for
that, a completely different code path).

Uses xgap's OWN env construction (harness.make_real_libero_env) -- this is
the point: validating that xgap's own harness reproduces the same
real-world result as the official `lerobot-eval` CLI (which used lerobot's
own `make_env`), since the eventual Oracle/World-model/Random experiments
need this harness for control lerobot-eval's CLI doesn't expose
(n_decision_points/exec_horizon/selection_unit -- see README "Design
constraints"). Policy loading and observation preprocessing reuse lerobot's
own public functions directly (make_policy, make_pre_post_processors,
make_env_pre_post_processors, preprocess_observation) -- the same ones
lerobot-eval's own rollout() calls internally (verified from
lerobot/scripts/lerobot_eval.py source, same as scripts/check_image_mirroring.py
already does successfully for one env.reset()).

Init-state selection is STANDARD order: episode_seed is passed directly as
harness.make_real_libero_env's demo_episode_index_within_task, matching
lerobot-eval's own init_states[episode_index % len(init_states)] convention
-- see harness.py's docstring for why this differs from demo replay's usage
of that same parameter.

KNOWN UNVERIFIED RISK, flagged rather than hidden: lerobot's
preprocess_observation()/env_preprocessor()/preprocessor() are designed for
lerobot's own BATCHED gym.vector.VectorEnv (see rollout()'s type hint in
lerobot_eval.py), but harness.make_real_libero_env returns a single,
UNBATCHED env (matching replay.py's existing usage -- plain (7,) actions,
unbatched obs dict, confirmed from that module's real Colab runs). This
module manually adds/removes a batch dimension of size 1 around the shared
preprocessing calls to bridge that gap. Not yet verified against a real
Colab run -- if the first run errors here, this bridging logic is the first
place to look, not the reused lerobot functions themselves.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .gripper_metrics import longest_close_run
from .harness import read_actual_control_mode
from .logging_schema import EpisodeRecord
from .metering import StageTimer


def build_policy_and_processors(
    *,
    checkpoint_path: str,
    task_suite_name: str,
    task_id: int,
    control_freq: int,
):
    """Build (policy, preprocessor, postprocessor, env_preprocessor,
    env_postprocessor) via lerobot's real `@parser.wrap()` config-construction
    path -- same mechanism scripts/check_image_mirroring.py already verified
    working end to end in Colab (real cfg -> real make_policy download ->
    real preprocessors), reused here instead of duplicated. Does NOT call
    lerobot's own make_env -- the env comes from harness.make_real_libero_env
    instead, per this module's whole point (see module docstring).
    """
    import sys

    from lerobot.configs import parser as lerobot_parser
    from lerobot.configs.eval import EvalPipelineConfig
    from lerobot.envs import make_env_pre_post_processors
    from lerobot.policies import make_policy, make_pre_post_processors

    @lerobot_parser.wrap()
    def _capture(cfg: EvalPipelineConfig) -> EvalPipelineConfig:
        return cfg

    saved_argv = sys.argv
    sys.argv = [
        "lerobot-eval",
        f"--policy.path={checkpoint_path}",
        "--env.type=libero",
        f"--env.task={task_suite_name}",
        f"--env.task_ids=[{task_id}]",
        "--eval.batch_size=1",
        "--eval.n_episodes=1",
        "--env.max_parallel_tasks=1",
        "--output_dir=/tmp/xgap_policy_rollout_cfg",
    ]
    try:
        cfg = _capture()
    finally:
        sys.argv = saved_argv

    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env, rename_map=cfg.rename_map)
    policy.eval()

    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)

    return policy, preprocessor, postprocessor, env_preprocessor, env_postprocessor


def _batch_observation(obs: dict) -> dict:
    """Add a leading batch dim of 1 to every leaf array in a raw harness env
    observation -- see module docstring, KNOWN UNVERIFIED RISK."""

    def _wrap(x):
        if isinstance(x, dict):
            return {k: _wrap(v) for k, v in x.items()}
        return np.asarray(x)[None, ...]

    return _wrap(obs)


def rollout_policy_episode(
    env: Any,
    policy: Any,
    *,
    preprocessor: Any,
    postprocessor: Any,
    env_preprocessor: Any,
    task_description: str,
    episode_seed: int,
    max_steps: int,
    task_id: str,
    task_suite: str,
    condition: str,
    control_mode: str,
    control_freq: int,
    checkpoint_name: str,
    checkpoint_hash: str,
    git_commit: str,
    config_file: str,
) -> EpisodeRecord:
    """Run ONE episode with the policy actually choosing actions (closed-loop,
    one policy.select_action() call per env step -- matches lerobot-eval's
    per-step behavior at n_action_steps=1; this project's own n_action_steps
    sweep already showed 1 vs 10 makes no success-rate difference for this
    checkpoint on libero_10, see README, so this simpler per-step form is
    used here rather than re-adding chunk-execution bookkeeping)."""
    import torch

    timer = StageTimer()
    t_episode_start = time.perf_counter()

    with timer.stage("reset"):
        obs, _info = env.reset(seed=episode_seed)

    actual_control_mode = read_actual_control_mode(env)
    policy.reset()

    success = False
    executed_actions: list[list[float]] = []
    executed_states: list[list[float]] = []
    n_steps = 0

    while n_steps < max_steps:
        with timer.stage("inference"):
            batched_obs = _batch_observation(obs)
            batched_obs["task"] = [task_description]
            batched_obs = env_preprocessor(batched_obs)
            batched_obs = preprocessor(batched_obs)
            with torch.inference_mode():
                action = policy.select_action(batched_obs)
            action = postprocessor(action)
        action_numpy = action[0].to("cpu").numpy()

        eef_pos = np.asarray(obs["robot_state"]["eef"]["pos"], dtype=np.float32)
        gripper_qpos = np.asarray(obs["robot_state"]["gripper"]["qpos"], dtype=np.float32)
        executed_states.append(np.concatenate([eef_pos, gripper_qpos]).tolist())
        executed_actions.append(action_numpy.tolist())

        with timer.stage("physics"):
            obs, _reward, terminated, truncated, info = env.step(action_numpy)
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
        environment_seed=episode_seed,
        condition=condition,
        action_chunk=executed_actions,
        state_chunk=executed_states,
        video_frame_steps=[],
        episode_success=success,
        rollout_length=n_steps,
        checkpoint_name=checkpoint_name,
        checkpoint_hash=checkpoint_hash,
        git_commit=git_commit,
        config_file=config_file,
        execution_time=execution_time,
        control_mode=control_mode,
        control_freq=control_freq,
        actual_control_mode=actual_control_mode,
        longest_close_run_steps=longest_close_run(executed_actions),
        **timer.as_fields(),
    )
