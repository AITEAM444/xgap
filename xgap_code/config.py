"""Experiment config loading. All experiment parameters live in configs/*.yaml -- no
hardcoded constants here or in scripts/."""

from __future__ import annotations

import multiprocessing
import os
from dataclasses import dataclass, field
from typing import Any

import yaml


def _resolve_parallel_envs(requested: int | str, cap: int) -> int:
    """`requested` is either a positive int or the literal string "auto" (derive from nproc)."""
    if requested == "auto":
        detected = multiprocessing.cpu_count()
        return max(1, min(detected, cap))
    requested = int(requested)
    if requested <= 0:
        raise ValueError(f"parallel_envs must be positive, got {requested}")
    return min(requested, cap)


@dataclass
class DemoReplayConfig:
    """Config for step 2: replay recorded demo actions through the eval harness.

    No policy is involved here -- this measures whether the environment, task
    init-state selection, and success detection are trustworthy before any
    policy diagnosis is attempted.
    """

    experiment_name: str
    # Local disk staging area (fast, per-episode writes land here immediately).
    # On Colab this is ephemeral VM disk (e.g. /content/...) -- wiped on session end.
    local_output_root: str
    task_suites: list[str]

    # Persistent store (Drive-mounted path). This is what resume checks against,
    # since local disk does not survive a Colab session restart. None => no sync,
    # local_output_root is treated as the persistent store too (used for local dev/testing).
    remote_output_root: str | None = None
    dataset_repo_id: str = "HuggingFaceVLA/libero"
    # task_ids[suite] omitted or null -> all tasks in that suite
    task_ids: dict[str, list[int] | None] = field(default_factory=dict)
    episodes_per_task: int = 1

    # Dual-run per user decision: both are executed and compared, the higher-success
    # one is adopted as the confirmed demo/env convention. This is NOT a sweep axis
    # (H4 in the policy sweep is separate, step 4) -- it only pins down which
    # convention the *environment and demo data* agree on before a policy is involved.
    control_modes: list[str] = field(default_factory=lambda: ["relative", "absolute"])

    # Defaults to the dataset's recorded fps (10.0, read from meta/info.json of
    # HuggingFaceVLA/libero), NOT lerobot's LiberoEnv default of 20. If replayed
    # actions were collected at 10Hz control_freq and we step the sim at 20Hz,
    # every action is applied at 2x the intended rate -- a distinct, plausible
    # bug source from control_mode. Logged explicitly per episode so this is
    # checkable after the fact, not just assumed correct.
    control_freq: int = 10

    # Must stay True: LiberoEnv ties init_state_id to episode_index, and replaying
    # demo actions only makes sense against the same init state the demo was
    # collected from. Asserted (not just defaulted) in harness.py.
    init_states: bool = True

    # Project-wide invariants (see README "Design constraints"). Unused by replay
    # itself (there is no candidate branching without a policy) but present now
    # so later conditions (Oracle/World-model/Random) share the same config shape.
    n_decision_points: int = 1
    exec_horizon: int = 1
    selection_unit: str = "chunk"  # step-level selection would break chunk-internal
    # structure once candidates exist -- kept as a labeled-but-unused field, not a magic string.

    parallel_envs: int | str = "auto"
    parallel_envs_cap: int = 8
    sync_every_n_episodes: int = 5

    # Instrumentation (step 2/3 diagnostics -- see README "instrumented rollout"). 0 =
    # disabled. Every N steps, env.render() is called; all sampled frames for an episode
    # are written as one <output_root>/videos/<episode_key>.mp4 (1 = every step, for a
    # smooth watchable video). Extra render() calls, so off by default; turn on for a
    # small run when you actually want to watch what happened.
    video_sample_every_n_steps: int = 0
    # Per-episode eef-position / gripper-qpos / gripper-action plot, saved under
    # <output_root>/videos/<episode_key>.png. Cheap (matplotlib on already-logged data,
    # no extra env calls) -- on by default.
    save_trajectory_plots: bool = True

    checkpoint_name: str = "N/A_demo_replay"
    checkpoint_hash: str = "N/A_demo_replay"
    git_commit: str = "unknown"

    random_seed: int = 0
    resume: bool = True

    def __post_init__(self):
        if not self.task_suites:
            raise ValueError("task_suites must be non-empty")
        if not self.control_modes:
            raise ValueError("control_modes must be non-empty")
        for m in self.control_modes:
            if m not in ("relative", "absolute"):
                raise ValueError(f"invalid control_mode '{m}', expected 'relative' or 'absolute'")
        if not self.init_states:
            raise ValueError(
                "init_states=False is not valid for demo replay: replaying demo actions "
                "against a random/default init state cannot validate the harness."
            )
        if self.selection_unit not in ("chunk", "step", "random"):
            raise ValueError(f"invalid selection_unit '{self.selection_unit}'")
        self.parallel_envs = _resolve_parallel_envs(self.parallel_envs, self.parallel_envs_cap)

    @classmethod
    def from_yaml(cls, path: str) -> "DemoReplayConfig":
        with open(path, encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f)
        known = {k for k in cls.__dataclass_fields__}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config fields in {path}: {sorted(unknown)}")
        return cls(**raw)


@dataclass
class InitStateSweepConfig:
    """Config for scripts/sweep_init_states.py: replay ONE demo episode's fixed
    action sequence against EVERY candidate init_state for its task.

    Different shape from DemoReplayConfig on purpose -- this isn't a matrix of
    tasks/episodes/control_modes, it's one fixed demo swept across LIBERO's own
    init_state indices, to test directly whether `within_task_index` (dataset_io.py)
    picks the init_state LIBERO actually recorded this demo from. See README
    "leading hypothesis: orientation" for why this sweep was proposed as a
    cheaper, more decisive test to run before extending state_chunk to orientation.
    """

    experiment_name: str
    local_output_root: str
    task_suite: str
    task_id: int

    remote_output_root: str | None = None
    dataset_repo_id: str = "HuggingFaceVLA/libero"
    # Which demo (by within_task_index, dataset_io.py's sorted-episode-index ordering)
    # to fix and sweep across init_states. 0 = the first/lowest-episode-index demo for
    # this task -- e.g. libero_10 task 0 -> global episode_index 8, see README.
    demo_within_task_index: int = 0

    control_mode: str = "relative"
    control_freq: int = 20  # see README "control_freq was wrong from the start"

    sync_every_n_episodes: int = 5

    checkpoint_name: str = "N/A_init_state_sweep"
    checkpoint_hash: str = "N/A_init_state_sweep"
    git_commit: str = "unknown"

    resume: bool = True

    def __post_init__(self):
        if self.control_mode not in ("relative", "absolute"):
            raise ValueError(f"invalid control_mode '{self.control_mode}'")

    @classmethod
    def from_yaml(cls, path: str) -> "InitStateSweepConfig":
        with open(path, encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f)
        known = {k for k in cls.__dataclass_fields__}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config fields in {path}: {sorted(unknown)}")
        return cls(**raw)
