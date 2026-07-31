# xgap

Week-1 scope: **decide whether SmolVLA behaves correctly on LIBERO**, before any
world-model or best-of-N work starts. No fixing — this repo diagnoses.

## Layout

```
xgap/
├── xgap_code/       our package (config, logging/resume store, harness, replay, dataset_io, metering)
├── configs/         yaml configs (one file per experiment; no hardcoded constants in code)
├── scripts/         CLI entry points -- the unit of execution, not notebook cells
├── tests/           pytest smoke tests against MockLiberoEnv -- run without mujoco/libero/lerobot
├── outputs/         <experiment_name>/{config.yaml, episodes.parquet, summary.csv, videos/, logs/}  (gitignored)
├── notebooks/       thin wrappers that only call scripts/
├── setup_colab.sh   idempotent env rebuild, run at the top of every Colab session
├── requirements.txt pinned deps
└── README.md
```

## Setup

```bash
bash setup_colab.sh
```

Safe to re-run. See the script's own comments for why `MUJOCO_GL=egl` must
additionally be set as the *first line of your notebook*, not just in the shell
script (Colab's `!` subshell env does not propagate into the kernel process).

### Dataset cache decision

**Status: not yet measured — do not trust a default here.** `setup_colab.sh`
currently points `LEROBOT_DATASET_CACHE` at local Colab disk (`/content/...`)
and `HF_HOME` (checkpoints/configs) at Drive. Before relying on this, run once
in Colab and record here:

- `du -sh` on the downloaded `HuggingFaceVLA/libero` dataset cache
- wall-clock time for a cold download vs. a warm local read vs. a warm Drive read
- verdict: local disk / Drive, and why

## Checkpoint under test

`HuggingFaceVLA/smolvla_libero` — see step-1 findings below. This model card
has no LIBERO-suite-specific info, so the diagnostic sweep (see plan) must
include at least one of `libero_object` / `libero_spatial` in addition to
`libero_10`, since we do not know which suite(s) it was trained on.

**Update:** `meta/tasks.parquet` in the presumed training dataset
(`HuggingFaceVLA/libero`, see below) lists exactly 40 task names, splitting
cleanly into 4 blocks of 10 that match the known task text for
`libero_10`/`libero_goal`/`libero_object`/`libero_spatial` respectively. This
is evidence (not proof — the checkpoint isn't provably trained on exactly this
dataset) that the checkpoint's training distribution spans all 4 standard
suites, not just `libero_10`. `configs/demo_replay.yaml` includes 3 of the 4
suites for this reason (`libero_goal` omitted only to keep the first Colab run
small).

## Step 1 findings (environment verification)

All facts below were read from the actual source (GitHub `huggingface/lerobot`
`main` branch and the HF Hub repo files for the checkpoint), not from memory.
Re-verify against the *installed* package version once `setup_colab.sh` has
actually run in Colab, since `main` can drift from the pinned release.

**Checkpoint `config.json` (HuggingFaceVLA/smolvla_libero):**

| field | value |
|---|---|
| `input_features` | `observation.images.image` (3,256,256), `observation.images.image2` (3,256,256), `observation.state` (8,) |
| `output_features` | `action` (7,) |
| `chunk_size` | 50 |
| `n_action_steps` | 1 |
| `normalization_mapping` | VISUAL: IDENTITY, STATE: MEAN\_STD, ACTION: MEAN\_STD |
| `resize_imgs_with_padding` | [512, 512] |
| `n_obs_steps` | 1 |

**`observation.images.image2` (wrist camera) IS declared as a required input.**
Any harness that only feeds `observation.images.image` is running the
checkpoint out-of-distribution for one entire modality (H2 is directly
testable against this).

**`n_action_steps=1` in the checkpoint config is a strong lead on H1.** This
value means the checkpoint's own config asks for a fresh policy call every
single env step (closed-loop, using only the first of 50 predicted actions
each time) — not open-loop execution of the full 50-step chunk. However,
lerobot's own published LIBERO reproduction (`pi05_libero_finetuned`) overrides
this at eval time (`--policy.n_action_steps=10`), so the shipped config value
is not proof of the *correct* value for this specific checkpoint — it must be
swept (H1), not assumed.

**Normalization stats (mean/std for state & action, per-dimension) live in
binary safetensors files** (`policy_preprocessor_step_5_normalizer_processor.safetensors`,
`policy_postprocessor_step_1_unnormalizer_processor.safetensors`), not
inspectable as text. These must be loaded in Python during the actual Colab
verification step to check `action` dim 6 (gripper) specifically for H3.

**lerobot's installed LIBERO integration** (`src/lerobot/envs/libero.py`,
`src/lerobot/envs/configs.py`, docs at `docs/source/libero.mdx`,
`docs/source/env_processor.mdx`):

- 5 task suites: `libero_spatial`, `libero_object`, `libero_goal`, `libero_90`, `libero_10` (= LIBERO-Long).
- Camera key mapping: `agentview_image -> image`, `robot0_eye_in_hand_image -> image2`. This is fixed by `camera_name_mapping` and swappable via config — directly usable for the H2 sweep condition ("image/image2 뒤바꿈").
- `observation.state` (8D) = `eef_pos (3) ++ quat2axisangle(eef_quat) (3) ++ gripper_qpos (2)`, built by `LiberoProcessorStep` (`ProcessorStepRegistry` name `"libero_processor"`), **not** inside the env itself. This matches the user's prior description exactly.
- **`LiberoProcessorStep` also flips every `observation.images.*` tensor 180° (`torch.flip(img, dims=[2,3])`)**, documented as matching a rotation convention baked into the `HuggingFaceVLA/libero` training dataset. This is a strong, easy-to-miss additional failure mode: if our harness feeds images to the policy without this flip (or flips when it shouldn't), the policy sees images in the wrong orientation relative to training — a plausible root cause for "reaches toward the object but fails to grasp," fully consistent with the reported symptom. **This must be added as an explicit checked condition in the instrumented rollout (step 3) and sweep (step 4), not assumed correct or incorrect.**
- Action space: `Box(-1, 1, shape=(7,))` = 6D eef delta + 1D gripper. `get_libero_dummy_action() == [0,0,0,0,0,0,-1]` — i.e. `-1` is the LIBERO no-op/settle value for the gripper dimension, a concrete reference point for H3's sign-convention check.
- `control_mode`: `"relative"` (default) or `"absolute"`, implemented as `robot.controller.use_delta = True/False` on env reset. Directly maps to H4.
- **Open question, not yet resolved from docs alone:** the `EnvConfig` dataclass in `configs.py` defaults `observation_height`/`observation_width` to **360**, while the checkpoint's declared `input_features` shape is **256×256**, and the checkpoint config separately carries `resize_imgs_with_padding: [512, 512]`. It is not yet confirmed from source alone whether/where the 360→256 (or →512) resize happens (policy-side vs. processor-side) or whether it's silently wrong by default. **Flagged for step 1's empirical follow-up in Colab: print actual tensor shapes at each pipeline stage (env output → env processor output → policy preprocessor output) rather than trust the docs.**

## Step 2 prep (this session — no Colab access, no lerobot/mujoco installed)

This session had no GPU/Colab and no LIBERO simulator available (checked, not
guessed — see below). Per the working decision made this session, scope was
narrowed to what's verifiable without a simulator: mock-based smoke tests of
the harness/logging/resume code, a real (non-simulated) statistical baseline
for H3 from the actual HuggingFaceVLA/libero dataset, and writing (but not yet
running for real) the demo-replay code for step 2 proper.

### Local environment check

`python -c "import mujoco"` / `import libero` / `import lerobot"` all fail
(`ModuleNotFoundError`) on this machine (Windows, Python 3.13.14). Per
instruction, no attempt was made to install them here — LIBERO is Linux-only
(confirmed in `docs/source/libero.mdx` and the `hf-libero` pip extra's
`sys_platform == 'linux'` marker) and this is a Windows dev box; the real run
happens in Colab. `pandas`, `pyarrow`, `pyyaml` were already available;
`pytest` and `huggingface_hub` were installed locally (lightweight, no
mujoco/GPU dependency, safe to add regardless of platform).

### Mock-based smoke tests (`tests/`, 17 tests, all passing)

Cover: config yaml parsing + validation (unknown-field rejection, invalid
`control_mode`/`selection_unit` rejection, `init_states=False` rejection,
`parallel_envs: auto` resolution/capping), the episode store (write/read
roundtrip, resume-skips-completed, **resume checks the remote/persistent store
not local staging** since local disk doesn't survive a Colab restart, batched
sync only at the configured threshold, partial-result tolerance), the full
replay loop against `MockLiberoEnv`, and the control_mode decision logic
(winner / both-low-suspect-init-state / exact-tie-inconclusive branches).

**Bug caught by these tests before it could burn Colab time:** `episode_key()`
built filenames like `demo_replay_relative__libero_10__libero_10:0__0.parquet`
— the `:` from `task_id="libero_10:0"` is invalid in Windows filenames (NTFS
reads `name:suffix` as an alternate-data-stream). The write call didn't raise;
it silently created an ADS invisible to a normal directory listing, so
`read_all()` returned zero rows. Fixed in `xgap_code/logging_schema.py`
(`episode_key` now also replaces `:` with `-`). Colab is Linux so this specific
bug wouldn't have reproduced there, but the class of bug (paths built from
raw task-suite/task-id strings) would have; worth keeping the test.

A second, harmless inconsistency surfaced while re-running the CLI manually:
demo-replay `resume` correctly reused stale results from an earlier buggy mock
run instead of recomputing them — correct resume behavior, but a reminder that
`outputs/<experiment>/` must be cleared when the *generation logic* changes,
not just the config.

### H3 baseline, measured from real data (`scripts/analyze_demo_actions.py`, `outputs/demo_action_stats/`)

Read 5 real demo episodes directly from `HuggingFaceVLA/libero` data shards
(2 `libero_10` episodes — tasks 3/4, multi-grasp long-horizon; 3 `libero_spatial`
episodes — tasks 32/33/39, single-grasp) via `huggingface_hub`, loading only
`action`/`observation.state` columns (no image decoding):

- `action[:,6]` (gripper) is **discrete in exactly `{-1.0, +1.0}`** in every
  sampled episode — never continuous.
- Every sampled episode **starts at `-1.0`**, matching LIBERO's own
  `get_libero_dummy_action() == [0,0,0,0,0,0,-1]` no-op value.
- `states[:,6] - states[:,7]` (finger separation, an openness proxy) starts
  near its max, and drops sharply ~15-20 simulation steps after the first
  `+1.0` command, then rises again after the matching release — i.e.
  **`-1 = open`, `+1 = close`**, with the physical gripper visibly lagging the
  discrete action switch by ~15-20 steps (actuator dynamics, not instant).
- This gives a concrete, checkable reference for step 3/4: a policy rollout
  whose `action[:,6]` never reaches `+1`, or whose sign is flipped relative to
  this convention, or whose `gripper_qpos` never visibly closes even when
  `action[:,6]` does flip, are three distinguishable failure signatures — not
  just "gripper looks wrong."
- Caveat: n=5 episodes, 2 of 4 suites. The `{-1,+1}`/`-1=open` finding is a
  hardware/environment convention (unlikely to vary by task), but re-run
  across `libero_goal`/`libero_object` too before treating this as exhaustively
  confirmed. Full per-dimension stats in `outputs/demo_action_stats/episodes.json`.

**Dead end, documented so it isn't re-attempted — generalized, not file-specific:**
what's dead is not "one stale metadata file," it's **hand-navigating the
dataset's parquet layout at all** (resolving which shard holds which episode,
reading chunk/file-index columns, scanning shards to find a task). That
approach was only ever an expedient fallback for this session (no lerobot
installed locally), and it already showed why it's fragile: `meta/episodes/*.parquet`'s
`data/chunk_index` / `data/file_index` columns (meant to map an episode to its
shard file) turned out to be **stale** for this dataset as currently hosted —
they claim episodes 0-14 all live in `chunk-000/file-000.parquet`; that file on
disk only actually contains episodes 0-2. There is no reason to expect the next
hand-rolled shortcut through this layout to be more trustworthy than that one
was.

**The living path is `LeRobotDataset` (lerobot's own dataset API), full stop.**
`xgap_code/dataset_io.py`'s real implementation goes through it, not through
any parquet-shard bookkeeping of ours. `scripts/analyze_demo_actions.py` is
the one exception, and it stays an explicit, self-contained, clearly-labeled
one-off (shard paths passed explicitly, never resolved automatically) that
produced this session's H3 baseline before lerobot was available — it is not
the pattern to extend or reuse for anything in steps 3+. Any future code that
needs demo data should call into `dataset_io.load_task_demo_episodes` (or
`LeRobotDataset` directly), never re-derive shard locations by hand.

### What's written but NOT yet verified for real

`xgap_code/harness.make_real_libero_env` and `xgap_code/dataset_io.py`'s
`LeRobotDataset`-based loader are written against `lerobot`'s `main`-branch
source (see Step 1 findings) but have never been executed — no lerobot install
was available this session. Before trusting a real (non-`--mock`) run of
`scripts/run_demo_replay.py` in Colab:

- confirm `LeRobotDataset(repo_id, episode_filter=...)` and
  `dataset.meta.episodes` / `dataset.hf_dataset` behave as documented in the
  installed version;
- confirm whether `HuggingFaceVLA/libero`'s per-task episode ordering matches
  LIBERO's own `task_suite.get_task_init_states()` ordering 1:1 (the
  `within_task_index` assumption in `dataset_io.py` / `harness.py`) — this is
  exactly the kind of mismatch the user flagged as a possible root cause
  independent of `control_mode` (see `decide_env_convention`'s
  `both_low_suspect_init_state` branch).

## Follow-up round: wiring verification, gripper-duration metric, plot, Colab assets

Four code changes made after the step-2-prep round above, plus the actual
Colab-facing execution assets (previously missing: a real notebook and a
smoke-scale config). All still validated only via `MockLiberoEnv` + pytest —
no lerobot/mujoco available this session (see "Local environment check").

**(a) control_mode plumbing verification.** A tie in demo-replay success rate
and a `control_mode` flag that never reached the simulator look identical
downstream (both conditions "behave the same") but are not the same finding —
one is a statement about physics, the other means we never tested different
physics at all. `harness.read_actual_control_mode(env)` reads
`robot.controller.use_delta` back from the real env's controller *after
reset* (not just the value we requested), logged as `actual_control_mode` on
every episode. `replay.verify_control_mode_wiring()` checks this — one
reading per requested mode is enough — and `scripts/run_demo_replay.py`'s
`run()` calls it as soon as it has one episode from each mode (i.e. after the
very first task's `episode_seed=0`), **aborting immediately with
`decision: "control_mode_not_wired"` before running anything else or
computing any success rate** if the two requested modes produced the same
actual controller state. `MockLiberoEnv` gained a `simulate_wiring_bug` test
hook so this abort path has real test coverage
(`tests/test_wiring_and_metrics.py`, `test_run_aborts_immediately_on_wiring_bug`).

**(b) Longest continuous gripper-close run, not just "did a close command
appear."** `gripper_metrics.longest_close_run()` measures the longest run of
consecutive `action[:,6] > 0` steps actually executed, logged as
`longest_close_run_steps` on every episode. Compared against
`DEMO_MIN_ACTUATION_LAG_STEPS = 15` (the lower bound measured from real demos
this session, see H3 baseline above): a rollout whose longest close run is
under 15 steps cannot have physically grasped, independent of whether the
sign convention is even correct — a third distinguishable failure mode
alongside "wrong sign" and "never commands close at all."

**(c) `action[:,6]` histogram, policy vs demo.**
`plots.plot_gripper_action_histogram(demo, policy, save_path)` overlays the
two distributions. The demo side is always the measured `{-1,+1}` two-spike
shape; if a real policy rollout's mass clusters near 0 instead, that reads as
a normalization bug on a dimension that should never be continuous (e.g.
MEAN_STD stats applied to a naturally bimodal signal) — visually distinct from
a sign flip, which still shows two spikes, just swapped. No real policy data
exists yet (step 3); `tests/test_wiring_and_metrics.py` exercises it against
synthetic demo-like and near-zero-clustered "policy" arrays to confirm the
plot renders and saves correctly.

**(d) Generalized the "dead end" note above** — it no longer singles out one
stale metadata file; it says hand-navigating this dataset's parquet layout at
all is the thing not to repeat, and `LeRobotDataset` is the only path to trust
going forward. (Section above has been rewritten in place, not appended.)

### Colab execution assets

- **`configs/demo_replay_smoke.yaml`** — 1 suite, 1 task, 1 episode, both
  `control_mode`s (2 episodes total). The *first* real (non-`--mock`) run
  should use this, not `configs/demo_replay.yaml`. Per instruction: code
  written against `lerobot` source without ever running it is expected to
  break on API mismatches first — that's plumbing work, not a science result,
  and should be shaken out at this minimal scale before widening to the full
  config. See "How to read the first real run" below.
- **`notebooks/run_replay.ipynb`** — thin by construction: cell 1 sets
  `MUJOCO_GL=egl` (before any other import, per `setup_colab.sh`'s own
  warning), cell 2 mounts Drive, cell 3 runs `setup_colab.sh` (which already
  logs `nproc`/`nvidia-smi`/`free -g`/library versions to
  `logs/env_meta.log` — that's the "log hardware in the first cell"
  requirement, satisfied by calling the script rather than duplicating its
  logging in notebook code), cell 4 runs `scripts/run_demo_replay.py` against
  the smoke config. No control flow lives in the notebook itself.

### How to read the first real (non-`--mock`) run

Per instruction, these two outcomes are NOT the same finding and must be
reported separately, not conflated:

- **Script crashes with a Python traceback** (`AttributeError`,
  `ImportError`, a `LeRobotDataset` constructor rejecting a kwarg, etc.) →
  this is an API mismatch between what `harness.py`/`dataset_io.py` assumed
  from reading `lerobot` source and what the actually-installed version
  provides. It is expected on the first run. Fix the plumbing, re-run the
  smoke config, repeat until it completes without crashing. **This is not
  evidence about SmolVLA or LIBERO** — do not record it as a hypothesis
  result.
- **Script completes and prints a decision** (whether
  `control_mode_confirmed`, `both_low_suspect_init_state`,
  `tie_inconclusive`, or the new `control_mode_not_wired`) → only once this
  happens is there an actual harness-fidelity result to report. What matters
  for the eventual verdict is the *gap* between the two `control_mode`
  success rates, not their absolute values — a harness that gets 60%/20% is
  more informative than one stuck at exactly 0%/0% either way.

Scale up (`episodes_per_task`, more `task_ids`, more suites) only after the
smoke config completes cleanly end to end.

### First real Colab run, first failure (plumbing, not science)

`setup_colab.sh` failed on `pip install -r requirements.txt`:

```
ERROR: Could not find a version that satisfies the requirement lerobot==0.6.1
ERROR: No matching distribution found for lerobot==0.6.1
```

This is exactly the "API mismatch, not a science result" case described above
— caught immediately, before any env/policy code even ran. Root cause,
confirmed via the PyPI JSON API (`https://pypi.org/pypi/lerobot/0.6.0/json`),
not guessed:

1. `0.6.1` was never published to PyPI — the version string in
   `requirements.txt` had been read off GitHub main's `pyproject.toml`, which
   is an unreleased dev version. Latest actual PyPI release is `0.6.0`.
2. More importantly: **the `libero` extra does not exist at all in `0.6.0`'s
   published metadata** (`requires_dist` has zero `libero`/`hf-libero`
   entries — checked directly, not assumed). LIBERO support is only on the
   unreleased main branch. Had the version number merely been off by a patch,
   `pip install lerobot[libero]==0.6.0` would have installed lerobot
   successfully while silently giving an environment with **no LIBERO
   integration at all** — a worse failure mode than the hard error we
   actually got, since it wouldn't have announced itself.

Fixed in `requirements.txt`: install directly from a pinned GitHub commit
(`0d0737ab57f27c05d7b35fcf27e701f6003a5f3a`, main as of 2026-07-30) instead of
PyPI, via `lerobot[libero] @ git+https://github.com/huggingface/lerobot.git@<sha>`.
Re-pin that commit deliberately if a newer lerobot is ever wanted; do not float
`main`.

## Design constraints encoded in this codebase (do not violate)

- `n_decision_points` (config field `n_decision_points`, default `1`) must be applied identically across Oracle / World-model / Random conditions — enforced in code, not by convention.
- `exec_horizon` (how many env steps run per policy call) and `selection_unit` (granularity at which candidates are compared) are separate config fields even before `selection_unit` has more than one implemented value. Step-level selection would break chunk-internal structure and produce a fake performance drop — this is a documented invariant, not a hypothetical.
- No monkey-patching `lerobot` or `libero`. If ever unavoidable, isolate in its own module with a comment explaining why.
- All experiment parameters live in `configs/*.yaml`. No hardcoded constants in `xgap_code/` or `scripts/`.
