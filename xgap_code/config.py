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


@dataclass
class PolicyRolloutConfig:
    """Config for scripts/run_policy_rollout.py: a real policy actually choosing
    actions (xgap's own harness env + lerobot's own policy/preprocessing pipeline
    -- see xgap_code/policy_rollout.py), NOT pre-recorded demo actions.

    Single suite by design (unlike DemoReplayConfig's multi-suite dict) -- this
    is the Gate-1/N=1 harness-parity check on libero_spatial specifically (see
    README "libero_spatial로 변경한다"), not a general multi-suite sweep.

    Init-state selection is STANDARD order (episode_seed directly, matching
    lerobot-eval's own convention), not dataset_io's demo-matching remap --
    see harness.make_real_libero_env's docstring.
    """

    experiment_name: str
    local_output_root: str
    task_suite: str
    task_ids: list[int]
    checkpoint_path: str

    remote_output_root: str | None = None
    episodes_per_task: int = 5

    # LiberoEnv's own episode-length cap. Not read from the env at runtime (no verified
    # API for that on xgap's own unbatched env, unlike lerobot's own VectorEnv, which
    # exposes env.call("_max_episode_steps") -- see lerobot_eval.py's rollout()) --
    # inferred instead from progress-bar output observed in earlier real lerobot-eval
    # runs this session ("Running rollout with at most N steps" climbing to ~520).
    # Print rollout_length per episode and compare against this if results look
    # truncated early.
    max_steps: int = 520

    control_mode: str = "relative"
    control_freq: int = 20  # see README "control_freq was wrong from the start"

    # Project-wide invariants -- unused by a single-policy rollout (no candidate
    # branching yet) but present now so later conditions (Oracle/World-model/Random)
    # share this config shape. See README "Design constraints".
    n_decision_points: int = 1
    exec_horizon: int = 1
    selection_unit: str = "chunk"

    checkpoint_hash: str = "unknown"  # HF revision, if pinned; not resolved automatically
    sync_every_n_episodes: int = 5
    git_commit: str = "unknown"

    resume: bool = True

    def __post_init__(self):
        if not self.task_ids:
            raise ValueError("task_ids must be non-empty")
        if self.control_mode not in ("relative", "absolute"):
            raise ValueError(f"invalid control_mode '{self.control_mode}'")
        if self.selection_unit not in ("chunk", "step", "random"):
            raise ValueError(f"invalid selection_unit '{self.selection_unit}'")

    @classmethod
    def from_yaml(cls, path: str) -> "PolicyRolloutConfig":
        with open(path, encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f)
        known = {k for k in cls.__dataclass_fields__}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config fields in {path}: {sorted(unknown)}")
        return cls(**raw)


@dataclass
class GateTwoDiversityConfig:
    """Config for scripts/run_gate2_diversity.py: does the policy actually
    propose meaningfully different candidates from the same observation, or
    does it collapse to a single mode (raised as a live concern after task 1
    showed 19/20 episodes with near-identical rollout_length/close_run
    values)? If candidates are indistinguishable, there is nothing for an
    Oracle/World-model/Random comparison to select between -- see README
    "Gate 2: candidate diversity".

    Branch points are read from ALREADY-RECORDED real episodes (a prior
    PolicyRolloutConfig run's own EpisodeStore output -- source_output_root/
    source_condition/source_episode_seed), not synthesized: each task's
    `source_episode_seed`'th episode's recorded action_chunk supplies the
    exact prefix needed to reach a real, actually-visited state via prefix
    replay (see harness.make_real_libero_env's docstring on STANDARD order,
    and README "Test 2: design change to prefix replay" for why this is
    deterministic).

    Diversity must come ONLY from the policy's own intrinsic stochasticity
    (predict_action_chunk(batch, noise=None) sampling fresh flow-matching
    noise internally each call -- verified from SmolVLAPolicy source, not
    guessed). No noise-scale/temperature knobs here on purpose -- inflating
    them would contaminate any later multimodality analysis.
    """

    experiment_name: str
    local_output_root: str
    task_suite: str
    task_ids: list[int]
    checkpoint_path: str

    remote_output_root: str | None = None

    # Where to read real recorded episodes (for prefix action_chunks) from --
    # a PREVIOUS PolicyRolloutConfig run's own output, not this config's own
    # local_output_root/remote_output_root (which is where THIS script's own
    # results get written).
    source_output_root: str = ""
    source_condition: str = "policy_rollout"
    source_episode_seed: int = 0

    # Where along the source episode's recorded trajectory to branch, as
    # fractions of its rollout_length. One state per fraction, per task.
    branch_step_fractions: list[float] = field(default_factory=lambda: [0.5])

    n_candidates: int = 64
    # How many of each candidate chunk's steps to actually EXECUTE (fresh env
    # + prefix replay + these steps) for the endpoint-variance metric -- the
    # expensive part (n_candidates fresh envs per branch point). Not the
    # full predicted chunk_size on purpose, to bound cost; this project's own
    # exec_horizon config concept, finally put to real use.
    exec_horizon: int = 10

    checkpoint_hash: str = "unknown"
    control_mode: str = "relative"
    control_freq: int = 20
    gripper_dim: int = 6

    resume: bool = True

    def __post_init__(self):
        if not self.task_ids:
            raise ValueError("task_ids must be non-empty")
        if not self.source_output_root:
            raise ValueError("source_output_root must be set (where to read prefix episodes from)")
        if not self.branch_step_fractions:
            raise ValueError("branch_step_fractions must be non-empty")
        if any(not (0.0 < f < 1.0) for f in self.branch_step_fractions):
            raise ValueError(f"branch_step_fractions must all be in (0, 1), got {self.branch_step_fractions}")
        if self.control_mode not in ("relative", "absolute"):
            raise ValueError(f"invalid control_mode '{self.control_mode}'")
        if self.n_candidates < 2:
            raise ValueError("n_candidates must be >= 2 to compute pairwise diversity")

    @classmethod
    def from_yaml(cls, path: str) -> "GateTwoDiversityConfig":
        with open(path, encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f)
        known = {k for k in cls.__dataclass_fields__}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config fields in {path}: {sorted(unknown)}")
        return cls(**raw)


@dataclass
class GateTwoOutcomeConfig:
    """Config for scripts/run_gate2_outcome.py: Gate 2 judged by episode
    OUTCOME (success/fail), not by action-diversity metrics.

    `GateTwoDiversityConfig`'s endpoint_variance metric is INVALIDATED for
    verdict purposes, not just ambiguous -- confirmed from real data, not
    a guess: `frac_commanding_close` (gripper_channel_distribution) moved
    0.53 at branch_fraction=0.2 to 0.85 at 0.5, i.e. candidates really do
    differ on WHEN they command the gripper to close, but
    `exec_horizon=10` is shorter than the gripper's own actuation lag
    (15-20 steps, `gripper_metrics.DEMO_MIN_ACTUATION_LAG_STEPS`) --
    a real timing difference between candidates has no way to show up as a
    physical endpoint difference within 10 steps. Endpoint-variance
    measurement stops once the branch_step_fractions=0.7 sweep point
    finishes; kept only as diagnostic evidence, not a Gate 2 input.

    Two design bugs in the first outcome-script attempt, both fixed here,
    not just relaxed:

    1. **`n_candidates` must actually be ~64, not 5.** A run of only 5
       trials that comes back all-same is NOT enough evidence to declare
       "no diversity that matters" -- with a true mix rate that just
       happens to be rare, 5 samples can easily miss it. Early-stopping
       the moment a mix IS observed (cheap, and already conclusive) is
       still correct and kept; the fix is raising the ceiling for the
       all-same case from 5 to 64 before concluding FAIL.
    2. **Source episodes must include real FAILURES, not only successes.**
       Branching at 50% into an already-successful demo near-guarantees
       the handed-off base policy also succeeds, regardless of which
       candidate was committed -- that tests "does resuming a good
       trajectory still work", not "does candidate choice matter", and
       says nothing about Gate 2. Failure-source branch points are the
       ones that actually matter: "was about to fail, but a good candidate
       flipped it to success" is the literal mechanism an Oracle curve
       measures. `source_episode_seeds` is therefore a dict keyed by
       `task_id` (not one scalar seed) -- pick a mix per task (~2 success +
       ~2 failure is a reasonable starting point), found via
       `scripts/list_source_episode_outcomes.py` against the real Gate-1
       recordings rather than guessed.

    At each (task_id, seed) branch point: sample ONE candidate, commit its
    first `exec_horizon` steps, hand off to the BASE POLICY (closed-loop,
    `policy_rollout.step_with_policy_until_done`) for the rest of the
    episode -- literally "Oracle" in miniature -- and record whether the
    episode succeeds. Repeat with fresh candidates (fresh env each time,
    per Test 2's binding design rule) up to `n_candidates`, stopping EARLY
    once both a success and a failure have been observed.

    Verdict per point: `mixed` if outcomes contain both a success and a
    failure. The `is_primary_signal` -- the one that actually decides Gate
    2 -- is `mixed AND source was a FAILURE episode`: this is the direct
    "Oracle can rescue a near-failure" evidence. A mixed result from a
    SUCCESS-source point is recorded but not decisive on its own (see
    module docstring problem 2). Gate 2 passes if `is_primary_signal` is
    true for at least one (task, seed) point; regardless of source
    outcome, any point where all candidates land the SAME result is
    evidence against that specific point (not automatically an overall
    Gate 2 failure by itself -- see README "Gate 2, corrected").

    `branch_fraction` is FIXED at 0.5 by direct decision (not derived from
    the now-invalidated endpoint-variance sweep) -- single value, not a
    list, since this script isn't swept itself.
    """

    experiment_name: str
    local_output_root: str
    task_suite: str
    task_ids: list[int]
    checkpoint_path: str

    remote_output_root: str | None = None

    source_output_root: str = ""
    source_condition: str = "policy_rollout"
    # task_id -> list of episode_seeds to branch from for that task -- a mix
    # of real recorded successes AND failures (see class docstring, problem
    # 2). Must have a non-empty entry for every task_id in `task_ids`.
    source_episode_seeds: dict[int, list[int]] = field(default_factory=dict)

    branch_fraction: float = 0.5

    # Steps of the sampled candidate's OWN chunk committed before handing off
    # to the base policy -- same role as GateTwoDiversityConfig's exec_horizon.
    exec_horizon: int = 10
    # Candidates tested per (task, source seed) branch point, early-stopping
    # once a mix of success/failure is observed. 64, not 5 (see class
    # docstring, problem 1) -- 5 trials all landing the same result is not
    # enough evidence to conclude "no diversity that matters".
    n_candidates: int = 64
    # Step budget for the base-policy handoff, counted FROM THE BRANCH POINT
    # (prefix_length + exec_horizon), not from episode step 0 -- lower than
    # Gate-1's 520 on purpose: success trajectories ran 66-121 steps, so 200
    # leaves real margin while cutting failure-episode cost substantially.
    # Counted-from-0 was a real bug caught before use: a FAILURE source can
    # have rollout_length=520, so branch_fraction=0.5 alone can already
    # exceed a from-0 cap of 200 before the handoff runs a single step (see
    # scripts/run_gate2_outcome.py's run_one_candidate_outcome docstring).
    max_steps: int = 200

    checkpoint_hash: str = "unknown"
    control_mode: str = "relative"
    control_freq: int = 20

    resume: bool = True

    def __post_init__(self):
        if not self.task_ids:
            raise ValueError("task_ids must be non-empty")
        if not self.source_output_root:
            raise ValueError("source_output_root must be set (where to read prefix episodes from)")
        if not self.source_episode_seeds:
            raise ValueError(
                "source_episode_seeds must be set -- a dict of task_id -> list of episode_seeds "
                "mixing real success AND failure episodes (see class docstring, problem 2)"
            )
        missing = [t for t in self.task_ids if t not in self.source_episode_seeds]
        if missing:
            raise ValueError(f"source_episode_seeds missing entries for task_ids {missing}")
        empty = [t for t, seeds in self.source_episode_seeds.items() if not seeds]
        if empty:
            raise ValueError(f"source_episode_seeds has empty seed lists for task_ids {empty}")
        if not (0.0 < self.branch_fraction < 1.0):
            raise ValueError(f"branch_fraction must be in (0, 1), got {self.branch_fraction}")
        if self.control_mode not in ("relative", "absolute"):
            raise ValueError(f"invalid control_mode '{self.control_mode}'")
        if self.n_candidates < 1:
            raise ValueError("n_candidates must be >= 1")

    @classmethod
    def from_yaml(cls, path: str) -> "GateTwoOutcomeConfig":
        with open(path, encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f)
        known = {k for k in cls.__dataclass_fields__}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config fields in {path}: {sorted(unknown)}")
        return cls(**raw)
