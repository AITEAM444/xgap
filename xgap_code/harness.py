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


def get_num_init_states(task_suite_name: str, task_id: int) -> int:
    """Number of candidate init states LIBERO has for (task_suite_name, task_id).

    `LiberoEnv.reset()` selects `init_states[episode_index % len(init_states)]`
    (src/lerobot/envs/libero.py) -- this is that same `len(init_states)`, via the
    exact function `LiberoEnv` itself calls internally
    (`lerobot.envs.libero.get_task_init_states`), not re-derived. Used by
    scripts/sweep_init_states.py to test every candidate index directly rather
    than continue guessing whether `within_task_index` picks the right one --
    see README "leading hypothesis: orientation" for why this sweep was proposed
    as a cheaper, more decisive test to run first.
    """
    from lerobot.envs.libero import get_task_init_states

    suite = _get_benchmark_suite(task_suite_name)
    return len(get_task_init_states(suite, task_id))


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
    """Construct a single (non-vectorized) LiberoEnv at a specific init_state.

    `demo_episode_index_within_task` is the index INTO LIBERO's own
    init-state ordering for this task (`task_suite.get_task_init_states()[idx
    % len(init_states)]`), NOT a dataset-global episode_index. Two different
    callers feed two different things through this same parameter:
      - Demo replay (replay.py): a value derived from dataset_io's
        within_task_index, trying to match a SPECIFIC recorded demo's actual
        init state. That mapping was investigated at length and abandoned as
        not generally solvable (see README "Mapping search abandoned") --
        not relevant outside demo replay.
      - Live policy rollout (policy_rollout.py): plain STANDARD order --
        episode_seed passed directly, 0/1/2/... -- matching
        `lerobot-eval`'s own `init_states[episode_index % len(init_states)]`
        convention. No demo, no dataset_io involved; this is the normal case.

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


def _snapshot_controller_state(controller) -> dict:
    """Shallow-copy only array/scalar attributes off a robosuite controller's
    `__dict__` -- e.g. integral terms, previous-command memory, ramp/filter
    state -- skipping object references (back-pointers to the robot/sim/env)
    so this never tries to deep-copy the whole simulator by accident. See
    get_sim_state's docstring for why this exists."""
    snapshot = {}
    for key, value in vars(controller).items():
        if isinstance(value, np.ndarray):
            snapshot[key] = value.copy()
        elif isinstance(value, (int, float, bool)):
            snapshot[key] = value
    return snapshot


def _restore_controller_state(controller, snapshot: dict) -> None:
    for key, value in snapshot.items():
        current = getattr(controller, key, None)
        if isinstance(current, np.ndarray) and isinstance(value, np.ndarray) and current.shape == value.shape:
            current[:] = value
        else:
            setattr(controller, key, value)


def get_sim_state(env):
    """Return the full underlying MuJoCo simulator state -- the position/
    velocity/time/actuator part in the same flattened format LIBERO's own
    `.hdf5` demo files store per-step (see dataset_io.py's module
    docstring, "if demo replay ever needs precise... init states again"),
    PLUS the contact solver's warm-start acceleration (`qacc_warmstart`)
    and the robot controller's own array/scalar state, captured separately.
    This is a DIFFERENT, more precise thing than LIBERO's `init_states`
    array (which only pins the very first frame of an episode) -- it
    captures the exact physics + controller state at any point.

    Why `qacc_warmstart` is captured separately: mujoco_py's
    `sim.get_state()` (`MjSimState`: time, qpos, qvel, act) does NOT
    include it. TESTED AND REJECTED as the cause of the noise this module
    was built to diagnose, though (scripts/test_state_restore_determinism.py:
    restoring it produced bit-identical results to not restoring it) --
    kept restored anyway since it's cheap and physically correct to do so,
    just not the actual explanation.

    Why the controller state is captured too: `restore_sim_state` only
    ever touched `env._env.sim` -- the robot's OSC_POSE controller
    (`env._env.robots[0].controller`, same object `read_actual_control_mode`
    reads) is a separate Python object with its own internal array/scalar
    state (integral terms, previous-command memory, ramp/filter buffers)
    that lives entirely outside `sim`. A restore that only resets physics
    leaves the controller's memory stale from whatever it was doing right
    before the restore -- consistent with the observed noise pattern
    (large initially, decaying over several steps as that stale memory
    gets overwritten by new commands). This is the current leading
    hypothesis, not yet confirmed -- see
    scripts/test_state_restore_determinism.py's result log in the README.

    Must be called after env.reset()/env.step(). Returns None if
    unavailable (e.g. MockLiberoEnv, which has no real physics to save).

    Returns {"flattened": ..., "qacc_warmstart": ... or None, "controller":
    {...} or None if no robots[0].controller found}.
    """
    inner = getattr(env, "_env", None)
    if inner is None:
        return None
    sim = getattr(inner, "sim", None)
    if sim is None:
        return None
    qacc_warmstart = getattr(sim.data, "qacc_warmstart", None)
    robots = getattr(inner, "robots", None)
    controller_state = None
    if robots:
        controller_state = _snapshot_controller_state(robots[0].controller)
    return {
        "flattened": sim.get_state().flatten(),
        "qacc_warmstart": None if qacc_warmstart is None else np.array(qacc_warmstart, copy=True),
        "controller": controller_state,
    }


def restore_sim_state(env, state) -> None:
    """Restore a state captured by get_sim_state(). Restores MuJoCo's
    physics state (qpos/qvel/time/act), `qacc_warmstart`, AND the robot
    controller's own array/scalar state (see get_sim_state's docstring for
    why each is handled) -- NOT necessarily any OTHER robosuite/LIBERO
    Python-level episode bookkeeping (step counters, cumulative reward,
    success-flag state). Whether that remaining gap matters in practice is
    exactly what scripts/test_state_restore_determinism.py exists to
    check empirically, not assume.
    """
    inner = getattr(env, "_env", None)
    if inner is None or getattr(inner, "sim", None) is None:
        raise ValueError("env has no underlying MuJoCo sim to restore state on (a real LiberoEnv is required)")
    sim = inner.sim
    sim.set_state_from_flattened(state["flattened"])
    if state.get("qacc_warmstart") is not None and hasattr(sim.data, "qacc_warmstart"):
        sim.data.qacc_warmstart[:] = state["qacc_warmstart"]
    if state.get("controller") is not None:
        robots = getattr(inner, "robots", None)
        if robots:
            _restore_controller_state(robots[0].controller, state["controller"])
    sim.forward()


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
