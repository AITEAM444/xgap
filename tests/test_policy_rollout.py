"""Smoke tests for the policy-rollout loop against MockLiberoEnv -- no
mujoco/libero/lerobot installed. Validates: the batching bridge
(_batch_observation), and rollout_policy_episode's loop mechanics (step
counting, success detection, resume-independent correctness) using a fake
policy/preprocessor stand-in instead of a real lerobot policy."""

from __future__ import annotations

import numpy as np
import torch

from xgap_code.harness import MockLiberoEnv
from xgap_code.policy_rollout import _batch_observation, rollout_policy_episode


def test_batch_observation_adds_leading_dim_recursively():
    obs = {
        "pixels": {"image": np.zeros((256, 256, 3), dtype=np.uint8)},
        "robot_state": {"eef": {"pos": np.array([1.0, 2.0, 3.0])}},
    }
    batched = _batch_observation(obs)
    assert batched["pixels"]["image"].shape == (1, 256, 256, 3)
    assert batched["robot_state"]["eef"]["pos"].shape == (1, 3)


class _IdentityProcessor:
    def __call__(self, x):
        return x


class _FakePolicy:
    """Always commands a full close (action[6]=1.0) on every step -- MockLiberoEnv
    reports success only if the FINAL step's gripper action is > 0."""

    def __init__(self):
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1

    def select_action(self, observation):
        action = torch.zeros((1, 7), dtype=torch.float32)
        action[0, 6] = 1.0
        return action


def test_rollout_policy_episode_success_path():
    env = MockLiberoEnv(max_steps=5, require_final_gripper_close=True)
    policy = _FakePolicy()

    record = rollout_policy_episode(
        env,
        policy,
        preprocessor=_IdentityProcessor(),
        postprocessor=_IdentityProcessor(),
        env_preprocessor=_IdentityProcessor(),
        preprocess_observation_fn=_IdentityProcessor(),
        task_description="put the bowl on the plate",
        episode_seed=0,
        max_steps=5,
        task_id="libero_spatial:0",
        task_suite="libero_spatial",
        condition="policy_rollout",
        control_mode="relative",
        control_freq=20,
        checkpoint_name="HuggingFaceVLA/smolvla_libero",
        checkpoint_hash="unknown",
        git_commit="deadbeef",
        config_file="configs/policy_rollout_libero_spatial_smoke.yaml",
    )

    assert record.episode_success is True
    assert record.rollout_length == 5
    assert policy.reset_calls == 1
    assert record.longest_close_run_steps == 5
    assert record.time_inference_s > 0
    assert record.time_physics_s > 0


class _NeverCloseFakePolicy:
    def reset(self):
        pass

    def select_action(self, observation):
        return torch.zeros((1, 7), dtype=torch.float32)  # gripper action stays 0, never > 0


def test_rollout_policy_episode_failure_path():
    env = MockLiberoEnv(max_steps=5, require_final_gripper_close=True)

    record = rollout_policy_episode(
        env,
        _NeverCloseFakePolicy(),
        preprocessor=_IdentityProcessor(),
        postprocessor=_IdentityProcessor(),
        env_preprocessor=_IdentityProcessor(),
        preprocess_observation_fn=_IdentityProcessor(),
        task_description="put the bowl on the plate",
        episode_seed=1,
        max_steps=5,
        task_id="libero_spatial:0",
        task_suite="libero_spatial",
        condition="policy_rollout",
        control_mode="relative",
        control_freq=20,
        checkpoint_name="HuggingFaceVLA/smolvla_libero",
        checkpoint_hash="unknown",
        git_commit="deadbeef",
        config_file="configs/policy_rollout_libero_spatial_smoke.yaml",
    )

    assert record.episode_success is False
    assert record.longest_close_run_steps == 0
