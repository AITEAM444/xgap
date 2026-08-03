"""Environment construction: real LIBERO env (via lerobot, imported lazily) and a
mock env for testing the harness without any simulator installed.

No monkey-patching of lerobot/libero -- this module only wraps the public API
documented in `lerobot.envs.libero` / `lerobot.envs.configs` (verified from
source; see README "Step 1 findings").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=None)
def _get_benchmark_suite(task_suite_name: str):
    """Lazy, memoized LIBERO suite lookup -- shared by make_real_libero_env and
    get_libero_task_language so a (suite, task_id) pair used repeatedly across
    episodes doesn't reconstruct the suite object every time."""
    from libero.libero import benchmark

    bench = benchmark.get_benchmark_dict()
    if task_suite_name not in bench:
        raise ValueError(f"Unknown LIBERO suite '{task_suite_name}'. Available: {sorted(bench.keys())}")
    return bench[task_suite_name]()


def get_libero_task_language(task_suite_name: str, task_id: int) -> str:
    """Return the natural-language task instruction LIBERO itself associates with
    (task_suite_name, task_id) -- e.g. "put the bowl on the plate".

    This is the exact string HuggingFaceVLA/libero's episode metadata stores per
    episode (`LeRobotDatasetMetadata.get_task_index()` looks up episodes by this
    string) -- it is NOT our own internal "suite:task_id" bookkeeping label (see
    run_demo_replay.py's `task_label`). Passing the wrong one of these two to
    `dataset_io.load_task_demo_episodes` is exactly the bug that produced
    `ValueError: The episode filter did not match any episode` on an actual
    Colab run -- see dataset_io.py's module docstring.
    """
    suite = _get_benchmark_suite(task_suite_name)
    return suite.get_task(task_id).language


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
    from lerobot.envs.libero import LiberoEnv

    suite = _get_benchmark_suite(task_suite_name)

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


def read_actual_control_mode(env) -> str | None:
    """Read back the control mode ACTUALLY wired into the env, not just the value
    we requested when constructing it.

    Why this matters: LiberoEnv.reset() sets `robot.controller.use_delta` based on
    `self.control_mode` (True for "relative", False for "absolute") -- see
    lerobot's `src/lerobot/envs/libero.py`. A plumbing bug (wrong kwarg name typo,
    a config field silently not forwarded, an env wrapper that reconstructs the
    env without passing control_mode through, etc.) could leave `use_delta`
    unchanged regardless of what we pass in, and the resulting demo-replay success
    rates would look like real physics evidence when they're actually noise from a
    control_mode that never reached the simulator. Must be called AFTER env.reset().

    Returns "relative" (use_delta=True), "absolute" (use_delta=False), or None if
    unavailable (env not yet reset, or a mock without a controller/reporting hook).
    """
    inner = getattr(env, "_env", None)  # real LiberoEnv's underlying OffScreenRenderEnv
    if inner is not None:
        robots = getattr(inner, "robots", None)
        if not robots:
            return None
        use_delta = getattr(robots[0].controller, "use_delta", None)
        if use_delta is None:
            return None
        return "relative" if use_delta else "absolute"
    # MockLiberoEnv test hook -- see its `reported_control_mode` property below.
    return getattr(env, "reported_control_mode", None)


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
    # What this mock instance was actually constructed with -- stands in for the real
    # env's `robot.controller.use_delta` so read_actual_control_mode() can be exercised
    # in tests without a simulator.
    control_mode: str = "relative"
    # Test hook: simulates a wiring bug where control_mode never reaches the env --
    # reported_control_mode always says "relative" regardless of what was requested.
    simulate_wiring_bug: bool = False
    _t: int = field(default=0, init=False)
    _last_gripper: float = field(default=-1.0, init=False)

    @property
    def reported_control_mode(self) -> str:
        return "relative" if self.simulate_wiring_bug else self.control_mode

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
        # eef drifts a little with step count so trajectory logging/plotting has something
        # non-degenerate to show in tests, without pretending to simulate real physics.
        pos = np.array([0.001 * self._t, 0.0, 0.0])
        return {
            "pixels": {"image": img, "image2": img},
            "robot_state": {
                "eef": {"pos": pos, "quat": np.array([0, 0, 0, 1.0]), "mat": np.eye(3)},
                "gripper": {"qpos": np.array([0.04, -0.04]), "qvel": np.zeros(2)},
                "joints": {"pos": np.zeros(7), "vel": np.zeros(7)},
            },
        }

    def render(self):
        return np.zeros((256, 256, 3), dtype=np.uint8)

    def close(self):
        pass
