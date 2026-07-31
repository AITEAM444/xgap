"""Environment construction: real LIBERO env (via lerobot, imported lazily) and a
mock env for testing the harness without any simulator installed.

No monkey-patching of lerobot/libero -- this module only wraps the public API
documented in `lerobot.envs.libero` / `lerobot.envs.configs` (verified from
source; see README "Step 1 findings").
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def make_real_libero_env(
    *,
    task_suite_name: str,
    task_id: int,
    demo_episode_index_within_task: int,
    control_mode: str,
    control_freq: int,
    observation_height: int = 256,
    observation_width: int = 256,
):
    """Construct a single (non-vectorized) LiberoEnv tied to a specific demo's
    recorded init state.

    `demo_episode_index_within_task` must be the index of this episode WITHIN
    its task's init-state ordering (`task_suite.get_task_init_states()[idx %
    len(init_states)]`), NOT a dataset-global episode_index. Whether
    HuggingFaceVLA/libero's per-task episode ordering matches LIBERO's own
    init-state ordering 1:1 is an assumption that must be verified empirically
    in Colab -- see xgap_code/dataset_io.py docstring. High replay success is
    itself supporting evidence; if control_mode and init_states are both
    correct yet replay success is low, this mapping is the next suspect.

    Imports lerobot/libero lazily so this module -- and MockLiberoEnv below --
    can be imported and unit-tested without either installed.
    """
    from libero.libero import benchmark
    from lerobot.envs.libero import LiberoEnv

    bench = benchmark.get_benchmark_dict()
    if task_suite_name not in bench:
        raise ValueError(f"Unknown LIBERO suite '{task_suite_name}'. Available: {sorted(bench.keys())}")
    suite = bench[task_suite_name]()

    return LiberoEnv(
        task_suite=suite,
        task_id=task_id,
        task_suite_name=task_suite_name,
        obs_type="pixels_agent_pos",
        observation_height=observation_height,
        observation_width=observation_width,
        init_states=True,  # required for demo replay -- see config.DemoReplayConfig
        episode_index=demo_episode_index_within_task,
        n_envs=1,
        control_mode=control_mode,
        control_freq=control_freq,
    )


@dataclass
class MockLiberoEnv:
    """Stands in for LiberoEnv in tests where mujoco/libero are not installed.

    Mimics the reset/step contract (obs dict with 'pixels' + 'robot_state',
    action Box(-1,1,(7,)), info['is_success']) so xgap_code.replay can be
    exercised end-to-end without a simulator. Physics are NOT simulated --
    success is a deterministic function of the injected action sequence,
    purely to validate harness/logging/resume code paths.
    """

    max_steps: int = 50
    require_final_gripper_close: bool = True
    _t: int = field(default=0, init=False)
    _last_gripper: float = field(default=-1.0, init=False)

    def reset(self, seed: int | None = None):
        self._t = 0
        self._last_gripper = -1.0
        return self._make_obs(), {"is_success": False}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (7,):
            raise ValueError(f"expected action shape (7,), got {action.shape}")
        self._t += 1
        self._last_gripper = float(action[6])
        terminated = self._t >= self.max_steps
        is_success = terminated and (not self.require_final_gripper_close or self._last_gripper > 0)
        info = {"is_success": is_success, "task": "mock_task", "task_id": 0, "done": terminated}
        return self._make_obs(), 0.0, terminated, False, info

    def _make_obs(self):
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        return {
            "pixels": {"image": img, "image2": img},
            "robot_state": {
                "eef": {"pos": np.zeros(3), "quat": np.array([0, 0, 0, 1.0]), "mat": np.eye(3)},
                "gripper": {"qpos": np.array([0.04, -0.04]), "qvel": np.zeros(2)},
                "joints": {"pos": np.zeros(7), "vel": np.zeros(7)},
            },
        }

    def close(self):
        pass
