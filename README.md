# xgap

Week-1 scope: **decide whether SmolVLA behaves correctly on LIBERO**, before any
world-model or best-of-N work starts. No fixing — this repo diagnoses.

Repo: https://github.com/AITEAM444/xgap (public — the code/configs here are
diagnostic tooling, no secrets or sensitive data; checkpoints/datasets/outputs
stay out of git via `.gitignore` and live on Drive/HF cache instead).

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

In Colab: open `notebooks/run_replay.ipynb` from GitHub (or clone by hand,
see below) and run its cells top to bottom -- it mounts Drive, clones/pulls
this repo into Drive, then runs `setup_colab.sh`. After the first clone,
picking up code changes is `git -C /content/drive/MyDrive/xgap pull` (the
notebook's clone/pull cell does this automatically every run); there is no
more manual re-uploading of files to Drive.

Manual/local:

```bash
git clone https://github.com/AITEAM444/xgap.git
cd xgap
bash setup_colab.sh
```

Safe to re-run. See the script's own comments for why `MUJOCO_GL=egl` must
additionally be set as the *first line of your notebook*, not just in the shell
script (Colab's `!` subshell env does not propagate into the kernel process).

### Dataset cache decision

**Decided: local disk (`/content`), not Drive** — via `HF_LEROBOT_HOME`
(lerobot's real dataset-cache env var; see "Ninth real Colab run" below for
the made-up `LEROBOT_DATASET_CACHE` this replaced). Briefly tried Drive
first, reasoning that the large one-time download needed to work around
`HuggingFaceVLA/libero`'s stale shard-location metadata (tens of GB) should
be persisted rather than repeated every session — but that download died
silently mid-transfer with no traceback and no user-initiated interrupt,
consistent with Colab's Drive FUSE mount being unreliable for sustained large
writes. Reverted to local disk: reliable completion matters more than
avoiding a repeat download, until there's a tested plan for syncing the
finished cache to Drive as one bulk copy afterward (not streaming the
download directly onto the Drive mount). Re-downloading tens of GB every
fresh Colab session is an accepted, known cost for now. `HF_HOME`
(checkpoints/configs, small) stays on Drive — that's a much smaller, more
FUSE-tolerant amount of data.

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

### Second real Colab run, second failure (also plumbing)

Next `pip install` attempt:

```
ERROR: Cannot install huggingface_hub<1.0 and >=0.24 and lerobot because these
package versions have conflicting dependencies.
ERROR: ResolutionImpossible
```

Root cause: our own `huggingface_hub>=0.24,<1.0` pin (added for
`scripts/analyze_demo_actions.py`) directly conflicted with lerobot's own
**core** dependency (not even the `libero` extra — its base install) of
`huggingface-hub>=1.6.0,<2.0.0`, read from `pyproject.toml`'s `dependencies`
list at the pinned commit. Fixed by widening our pin to `>=1.6.0,<2.0.0` to
match. Low-risk: this codebase already exercises `huggingface_hub` 1.16.4
locally (`scripts/analyze_demo_actions.py`'s real run that produced the H3
baseline above used this exact version), so the API surface we depend on
(`hf_hub_download`) is already confirmed working on a 1.x release, not just
assumed compatible.

### Third real Colab run, third failure (still plumbing, not us this time)

`pip install` succeeded (wheels for `bddl`, `robomimic`, `hf-egl-probe`,
`egl_probe` all built), but the setup verification step then crashed:

```
Do you want to specify a custom path for the dataset folder? (Y/N): Traceback...
  File ".../libero/libero/__init__.py", line 104, in <module>
    answer = input(...)
EOFError: EOF when reading a line
```

Root cause, read from the actual `hf-libero` source
(`huggingface/lerobot-libero`, the fork lerobot's `libero` extra installs),
not guessed: `libero/libero/__init__.py` runs an interactive
`input("Do you want to specify a custom path for the dataset folder? (Y/N): ")`
prompt the first time it's imported, gated purely on
`if not os.path.exists(config_file)` (`config_file` = `$LIBERO_CONFIG_PATH/config.yaml`,
default `~/.libero/config.yaml`) — there is no env var or flag to suppress it.
In any non-interactive context (this script, a Colab cell run as a
subprocess, CI) that `input()` immediately hits EOF and crashes.

Fixed in `setup_colab.sh` (new step 3, "Pre-seed the `libero` package's own
config.yaml"): before ever importing `lerobot.envs.libero` (which transitively
imports `libero.libero`), a small Python snippet writes that config file
itself, replicating `get_default_path_dict()`'s exact default-path logic from
the actual source. It only imports the *outer* `libero` package to locate the
install directory (confirmed empty `__init__.py` — does not execute
`libero/libero/__init__.py`, so this is safe and doesn't trigger the prompt
itself). `LIBERO_CONFIG_PATH` is set to local disk (`/content/xgap_libero_config`),
not Drive, since the paths it contains are only valid for the current
session's site-packages layout.

### Fourth real Colab run, fourth failure (our bug, in the real-data path)

Setup and the LIBERO env import both passed this time — the actual demo
replay logic ran and crashed:

```
ds = LeRobotDataset(repo_id, episode_filter=lambda ep: ep.get("tasks", [None])[0] == task_name)
ValueError: The episode filter did not match any episode. Make sure the
filter and episodes list are valid and compatible.
```

Root cause, confirmed by reading lerobot's actual pinned-commit source
(`src/lerobot/datasets/lerobot_dataset.py`, `dataset_metadata.py`) rather than
guessing again: `episode_filter` is evaluated against a per-episode metadata
dict keyed by `task_index` (an int) — fields are `task_index`, `episode_index`,
`length`, `from_timestamp`, `to_timestamp`. There is no `"tasks"` key at all.
`dataset_io.py`'s original `ep.get("tasks", [None])[0] == task_name` therefore
always fell through to the `.get()` default and never matched anything,
guaranteed to fail on every real run regardless of what `task_name` was.

A second, related bug in the caller: `run_demo_replay.py` was passing our own
internal bookkeeping label (`task_label`, e.g. `"libero_10:0"`) as `task_name`
— but the dataset's episodes are keyed by LIBERO's actual natural-language
task instruction (e.g. `"put the bowl on the plate"`). Even with the
`task_index` fix, the wrong string would still have matched nothing.

Fixed both, grounded in source, not guesswork:

- `dataset_io.load_task_demo_episodes` now resolves `task_name -> task_index`
  via `LeRobotDatasetMetadata.get_task_index()` (a real method, read from
  source), then filters on `ep["task_index"]`. It also now reads frames
  directly off the filtered `LeRobotDataset` (`ds[i]["action"]` /
  `ds[i]["observation.state"]`, torch tensors converted to numpy) instead of a
  second `hf_dataset.filter()` call that assumed HF `datasets` API shapes we
  hadn't confirmed either.
- `harness.get_libero_task_language(suite, task_id)` (new) returns
  `task_suite.get_task(task_id).language` — LIBERO's own instruction string —
  and `run_demo_replay.py` now passes *that* to `load_task_demo_episodes`,
  keeping `task_label` for our own logging only.

This one was on us, not a lerobot/libero surprise — worth naming as such
rather than filing it next to the previous three.

### Fifth real Colab run, fifth failure (env var propagation, an architecture bug)

The LIBERO dataset-path prompt (third failure, above) came back — but this
time inside `scripts/run_demo_replay.py`, in a cell run *after*
`setup_colab.sh` had already passed its own verification step cleanly.

Root cause: `setup_colab.sh`'s `export HF_HOME=...` / `LEROBOT_DATASET_CACHE`
/ `LIBERO_CONFIG_PATH` only affect *that script's own* subprocesses (the pip
install, its own verification import) — they die with the script. A separate
`!python scripts/run_demo_replay.py` cell is a brand-new subprocess that never
inherited them, so `LIBERO_CONFIG_PATH` was unset there, `libero.libero`
fell back to its own default (`~/.libero/config.yaml`) instead of the file
`setup_colab.sh` had pre-seeded at `/content/xgap_libero_config/config.yaml`,
found nothing there, and prompted again. The exact same gap almost certainly
also silently defeated `HF_HOME` (checkpoint caching to Drive) and
`LEROBOT_DATASET_CACHE` in every real run so far — this wasn't just a LIBERO
quirk, it was a hole in the whole env-var design.

Why `MUJOCO_GL` never had this problem: it's set via `os.environ` directly in
the notebook's own Python kernel (cell 1), and the kernel process's environ
*is* inherited by every subsequent `!`-cell subprocess for the rest of the
session — unlike a bash `export` inside one `!bash script.sh` invocation,
which isn't.

Fixed by extending that same mechanism to the rest of the vars, without
duplicating path defaults in two places: `setup_colab.sh` now writes its
*resolved* values (after defaults/overrides are applied) to
`/content/.xgap_env`, and `notebooks/run_replay.ipynb` gained a new cell,
right after the `setup_colab.sh` cell, that reads that file into the kernel's
`os.environ`. `setup_colab.sh` stays the single source of truth for the path
logic; the notebook just adopts whatever it resolved.

### Sixth real Colab run, sixth failure (a self-inflicted correction to the fourth)

Right after the env-var fix above, the *next* run hit:

```
KeyError: 'task_index'
```

This was a regression from the fourth-run fix itself: the `episode_filter`
docstring in `lerobot_dataset.py` claims per-episode rows carry a
`task_index` field, and that was taken at face value instead of cross-checked
against what this session had *already independently confirmed* by
downloading real parquet shards directly (see the H3 baseline section above,
and `outputs/demo_action_stats/`): the raw episode metadata has a `"tasks"`
key (`list[str]`), never `"task_index"`. Reading `load_episodes()`'s actual
implementation (`src/lerobot/datasets/io_utils.py`) confirms it: that
function loads the raw episodes parquet and drops only `stats/*` columns —
`"tasks"` survives untouched. The docstring is simply stale or describes a
different code path; it should not have outranked data this session had
already verified with its own hands. Filtering is back to `ep["tasks"][0] ==
task_name` — the *actual* bug in the fourth run was never the key name, only
the comparison value (our internal `task_label` instead of LIBERO's real
instruction string) — the `task_index` detour was introduced by trusting the
wrong source over already-verified data.

**Lesson applied going forward:** when a library's docstring and this
project's own already-verified empirical data disagree, the empirical data
wins — re-derive from source code (not comments/docstrings) to arbitrate,
rather than deferring to whichever was read most recently.

### Seventh real Colab run, seventh failure (a second regression from the same fourth-run fix)

Immediately after the sixth-run fix: the dataset downloaded successfully
(1.69 GB), then:

```
ValueError: need at least one array to stack
```

Also a self-inflicted regression, in the same code touched by the fourth-run
fix: `frames_by_episode` was pre-seeded with keys from `ds.episodes`
(`filter_episodes()`'s sorted, dataset-global episode indices), and only
frames whose `episode_index` matched one of those pre-seeded keys were kept.
There is no source-confirmed guarantee that the *filtered* `LeRobotDataset`'s
own `__getitem__` reports `episode_index` in that same numbering (as opposed
to, say, a compacted 0..N-1 numbering local to the filtered subset) — if it
doesn't, every frame gets silently dropped and `np.stack([])` on an empty list
raises exactly this error.

Fixed in `dataset_io.py` by building the episode grouping purely from
`episode_index` values actually observed on `ds[i]`, with no cross-referenced
assumption about a second accessor (`ds.episodes`) using the same numbering;
`episode_indices` is now derived from the grouping's own keys, not from
`ds.episodes` separately. Also added an explicit error (rather than a bare
`ValueError` from inside `numpy`) if the grouping ends up empty, so a future
occurrence is immediately distinguishable from "everything worked, 0 results"
via message text alone.

**Both the sixth and seventh failures were regressions introduced by the same
fourth-run fix**, not new lerobot/libero surprises — the underlying lesson is
the same one stated above: verify each new assumption (a docstring, an
unconfirmed accessor) against this session's own already-verified ground
truth before shipping it, rather than patching forward on unconfirmed
assumptions between Colab round-trips.

### First successful end-to-end run

After the seventh fix, `configs/demo_replay_smoke.yaml` (1 task, 1 episode,
both `control_mode`s) completed with no traceback:

```json
{
  "decision": "both_low_suspect_init_state",
  "mode_summary": {
    "relative": {"n_episodes": 1, "success_rate": 0.0},
    "absolute": {"n_episodes": 1, "success_rate": 0.0}
  }
}
```

This is the first real result, not a plumbing failure — but `n=1` per
`control_mode` is too small to act on (a single episode failing doesn't
distinguish "genuinely both broken" from "unlucky draw"). Two things were
fixed/changed in response, not to the finding itself, just to get a
trustworthy enough sample to interpret it:

- Both configs had `remote_output_root: null` — this run's results only ever
  landed on ephemeral `/content` disk and would not have survived a session
  restart. Pointed both at real Drive paths.
- `demo_replay_smoke.yaml`'s `episodes_per_task` raised `1 -> 5`.

### Eighth real Colab run, eighth failure (another regression, same root cause pattern)

Right after bumping `episodes_per_task`, the next run crashed:

```
IndexError: list index out of range
  demo_actions = demo_episodes[episode_seed].actions
```

Root cause: `_run_one_episode` called `load_task_demo_episodes(...,
max_episodes=cfg.episodes_per_task)` fresh for *every single*
`(episode_seed, control_mode)` pair, and treated `cfg.episodes_per_task` as a
guarantee of how many episodes would come back, not a request — if a task
happens to have fewer demo episodes in the dataset than `episodes_per_task`
asks for, `demo_episodes[episode_seed]` goes out of range for the later
seeds. (This was also simply wasteful regardless of the bug: re-filtering and
re-fetching the same task's episodes on every one of `episodes_per_task *
len(control_modes)` calls instead of once.)

Fixed in `scripts/run_demo_replay.py`: `load_task_demo_episodes` is now
called once per `(suite, task_id)`, before the episode-seed loop, and the
loop range is `min(cfg.episodes_per_task, len(demo_episodes))` with an
explicit printed note when the dataset has fewer episodes than requested,
instead of silently crashing later.

### Ninth real Colab run: a real bug, in the dataset/library, not our code

The eighth-run fix printed something that shouldn't have been possible:

```
[xgap] task 'put both the alphabet soup and the tomato sauce in the basket'
(libero_10:0) has only 2 demo episode(s) in HuggingFaceVLA/libero
```

Checked directly against the full 1693-episode metadata (downloaded earlier
this session, see the H3 baseline section): every one of the dataset's 40
tasks has between 29 and 50 episodes (mean ~42); this exact task has **33**.
"2" was not a real data characteristic — it was `dataset_io.py` silently
under-counting. Rather than guess a fourth time, a small diagnostic script
was run directly against the real, already-cached dataset to see what was
actually happening at each stage:

```
ds.episodes (filter_episodes result): [8, 13, 26, 39, 69, 71, 77, 79, 92, ...]  (33 total)
len(ds.episodes): 33
len(ds) [total frames]: 581
unique episode_index values actually seen on frames: [13, 26]
```

So `filter_episodes()` correctly identifies all 33 episodes at the metadata
level — the bug is downstream, in how `LeRobotDataset` materializes their
actual frame data.

**Root cause, confirmed from lerobot's actual source, not guessed:**
`LeRobotDatasetMetadata.get_data_file_path()`
(`src/lerobot/datasets/dataset_metadata.py`) resolves which shard file an
episode's data lives in via `meta/episodes/*.parquet`'s `data/chunk_index` /
`data/file_index` columns — **the exact same columns already found to be
stale for `HuggingFaceVLA/libero` earlier this session** (episode_index 8's
row claims `data/chunk-000/file-000.parquet`; that file's real contents are
only episodes 0-2 — see the "dead end" note above).
`DatasetReader.get_episodes_file_paths()`
(`src/lerobot/datasets/dataset_reader.py`) uses this to build the
`allow_patterns` list for a *selective* `snapshot_download` whenever
`LeRobotDataset` is constructed with `episode_filter`/`episodes` set — so for
most of the 33 matched episodes, the wrong (or an incomplete) shard file gets
fetched, and that episode's frames never materialize. Only episodes 13 and
26 happened to end up with usable data, most likely as an accidental
side-effect of whatever broader (also metadata-driven, also imprecise) file
set actually got downloaded.

**This is not our bug, and not fixable by changing our own filtering logic**
— it's a real defect either in `HuggingFaceVLA/libero`'s converted metadata
or in how lerobot's selective downloader trusts it. Per this project's own
rule against monkey-patching `lerobot`/`libero`, the fix is isolated and
explicit rather than patched into third-party code:
`xgap_code/dataset_io.ensure_full_dataset_cached()` force-downloads the
dataset's entire `data/` (and `meta/`) directory via `snapshot_download`
*directly into the same cache directory lerobot's own downloader reads from*
(`HF_LEROBOT_HUB_CACHE`, from `lerobot.utils.constants` — not a path we
invent), before `LeRobotDataset` is ever constructed with a filter. Whichever
(possibly wrong) files lerobot's own logic later asks for, they're already
present locally, so nothing is silently missing.

This surfaced a second, independent bug: `setup_colab.sh` had been exporting
a **made-up `LEROBOT_DATASET_CACHE` variable that lerobot never reads at
all** — it silently had zero effect since it was introduced. The real
variable is `HF_LEROBOT_HOME` (`lerobot.utils.constants`, defaults to
`$HF_HOME/lerobot`). Fixed, and — given the workaround above requires
downloading most/all of the dataset (tens of GB) regardless of which few
episodes are actually used — pointed at Drive rather than local `/content`,
reversing the earlier "local disk for read performance" decision: paying a
large download once and persisting it clearly beats repeating it every fresh
Colab runtime. `configs/demo_replay_smoke.yaml`'s docstring and
`notebooks/run_replay.ipynb` now both call out that the first real run pays
this one-time cost.

**Asked the user before proceeding** given the real time/bandwidth cost
(chunk-000 alone ~30GB, full dataset ~50GB at the observed ~45MB/s transfer
rate) — for a project explicitly building toward a paper, chose the full
dataset over a partial download, since `configs/demo_replay.yaml` already
plans to exercise `libero_spatial` (whose episodes extend into the dataset's
second chunk) and paying the larger cost once now avoids a second forced wait
later.

### Tenth real Colab run: the Drive-cache decision above didn't survive contact

The full-dataset download (now targeting Drive, per the decision above) died
silently partway through — no Python traceback, no `KeyboardInterrupt`, and
the user confirmed they hadn't pressed anything. A process dying with no
traceback at all (not even an interrupt handler running) points at something
outside Python killing it, not a normal exception. The leading suspect:
Colab's Drive FUSE mount is well known to be unreliable for sustained large
writes (tens of GB), separate from anything in our own code. Reverted
`HF_LEROBOT_HOME` to local disk (`/content/xgap_hf_lerobot_cache`) — see the
"Dataset cache decision" section above, now updated to match. Repeating the
large download every fresh session is an accepted cost until there's a
tested plan for a post-download bulk sync to Drive (copying already-complete
local files over, rather than streaming the download directly onto the FUSE
mount).

### Eleventh: an out-of-memory kill, and abandoning `LeRobotDataset` for frame access

Right after the local-disk revert above, the download died again — this time
with a clear cause: **system RAM exhaustion**, not disk/network. Unlike the
Drive-FUSE death, `snapshot_download` writing to local disk shouldn't itself
be RAM-hungry; the likely culprit is `LeRobotDataset` materializing full rows
(including the embedded image columns — this dataset stores images directly
inside the data parquet files, confirmed earlier this session, no separate
video files) for far more of the ~273k-frame dataset than our per-episode
access pattern actually needs.

Rather than patch this a third time, `xgap_code/dataset_io.py` was rewritten
to stop using `LeRobotDataset` for frame access entirely (its docstring now
documents this decision as CORRECTION 5, with the full chain of five prior
corrections above it). `LeRobotDatasetMetadata` is still used — its
metadata-level `get_task_index()` / `filter_episodes()` have been correct in
every single run so far, only the per-episode *file lookup* built on top of
it (`get_data_file_path()`, see the ninth-run section above) was broken.
Everything past "which episodes does this task have" is now handled by two
new functions:

- `list_shard_paths()` — lists the dataset's actual shard files from the Hub
  API directly, not the stale `data/chunk_index`/`data/file_index` metadata
  columns.
- `scan_shards_for_episodes()` — downloads shards **one at a time**, stopping
  as soon as every target episode is found (not the whole dataset), and reads
  **only** the `action`/`observation.state`/`episode_index`/`frame_index`
  columns via `pyarrow` column projection — the embedded image columns are
  never decoded. This bounds peak memory to roughly one shard's numeric data,
  regardless of dataset size, and is the same pattern
  `scripts/analyze_demo_actions.py` already used successfully this session
  (see the H3 baseline section), now generalized and given real test coverage
  (`tests/test_dataset_io.py`, 4 tests against a synthetic fixture shard —
  the one part of this whole dataset-loading saga that didn't need another
  blind Colab round-trip, since it only needs `huggingface_hub` + `pyarrow`,
  both available locally).

Also corrected in passing: the ninth-run section above reasoned that
`libero_spatial`'s episodes "extend into the dataset's second chunk" — false.
Checked directly via `HfApi().list_repo_files()`: the dataset has exactly
**377 shard files, all under `data/chunk-000/`** — no `chunk-001` exists at
all, despite `total_episodes=1693` exceeding `info.json`'s
`chunks_size=1000`. That assumption doesn't change anything now (nothing here
still depends on chunk numbering), but it's worth not repeating.

One more risk this surfaced and fixed: `scan_shards_for_episodes`'s own
`hf_hub_download` calls would, by default, cache under `HF_HOME` — which is
deliberately a Drive path (for small checkpoint/config caching). Left alone,
that's the exact same Drive-FUSE risk that just killed a run, through a
different code path. Fixed by passing `cache_dir` explicitly, resolved via
`tempfile.gettempdir()` (always local) rather than a new env var that could
suffer the same cross-cell-propagation issues already hit twice this session.

### Twelfth: not a hang, `max_episodes` was applied too late

The next run "looked stuck" after a couple of shard downloads completed. It
wasn't hung — `load_task_demo_episodes` was calling `scan_shards_for_episodes`
with the task's **full** episode set from `filter_episodes()` (e.g. all 33+),
and only truncating to `max_episodes` (5, for the smoke config) *after* the
scan returned. `scan_shards_for_episodes` stops as soon as every episode it
was *asked* for is found -- asking for 33 when only 5 are needed means it
correctly keeps working, just on ~28 episodes nobody needed yet, downloading
far more shards than necessary. Fixed: cap to `max_episodes` **before**
calling `scan_shards_for_episodes`, not after, so it stops as soon as the
actually-needed subset is found.

## First real science result: demo replay fails, 0/10, both control modes

`configs/demo_replay_smoke.yaml` (1 task, 5 episodes, both `control_mode`s)
finally ran end to end and produced a real (not plumbing-broken) result:

```json
{
  "decision": "both_low_suspect_init_state",
  "mode_summary": {
    "relative": {"n_episodes": 5, "success_rate": 0.0},
    "absolute": {"n_episodes": 5, "success_rate": 0.0}
  }
}
```

Per-episode telemetry (`rollout_length`, `longest_close_run_steps`,
`actual_control_mode`, already logged by design):

| condition | seed | success | rollout_length | longest_close_run |
|---|---|---|---|---|
| absolute | 0 | False | 285 | 87 |
| absolute | 1 | False | 296 | 81 |
| absolute | 2 | False | 296 | 81 |
| absolute | 3 | False | 288 | 82 |
| absolute | 4 | False | 292 | 92 |
| relative | 0 | False | 285 | 87 |
| relative | 1 | False | 296 | 81 |
| relative | 2 | False | 296 | 81 |
| relative | 3 | False | 288 | 82 |
| relative | 4 | False | 292 | 92 |

**`rollout_length` and `longest_close_run_steps` are identical between
`relative` and `absolute` per seed.** This is not evidence that `control_mode`
had no physical effect -- both fields are artifacts of the replay loop itself:
`rollout_length` is just "ran out of recorded demo actions" (LIBERO's own
`done`/`is_success` termination never fired in either condition, for any of
the 10 episodes), and `longest_close_run_steps` is computed from the *injected*
action array, which is identical in both conditions by construction (same
demo, same actions, only the pose-control interpretation differs).

**What the numbers do show:** every episode replayed the full recorded demo
(200-300 steps, no crash, no early termination) with a normal-looking
open→close→open gripper command pattern (`longest_close_run_steps` 81-92,
well above the ~15-20 step actuation lag measured from real demos) — the
replayed script looks structurally like a complete, well-formed demo. Yet
`episode_success` is `False` every time, **including when literally replaying
the dataset's own recorded ground-truth actions.**

This is exactly the pre-declared stop condition from the start of this
project: *"이게 낮으면 정책 이전에 하니스가 깨진 것이고 이후 진단은 전부
무의미하다"* (if this is low, the harness is broken before the policy is even
involved, and everything downstream is meaningless). No policy has been run
at any point in this investigation — this result is entirely about whether
our environment/harness reproduces the conditions the demo was recorded
under, and the answer right now is no. **Step 3 (policy rollout) is on hold
until this is resolved.**

Leading hypothesis, unchanged from when it was first flagged in
`harness.make_real_libero_env`'s docstring at the very start of this session:
`within_task_index` (this project's own dataset-global-episode-index-sorted
enumeration, used as `LiberoEnv`'s `episode_index` to select
`task_suite.get_task_init_states()[idx % len(init_states)]`) may not
correspond 1:1 with LIBERO's own original init-state ordering for this task.
A structurally-normal-looking replayed trajectory that still misses is the
expected signature of "started from the wrong physical configuration, then
blindly replayed the right relative motions anyway" — and matches the
original reported symptom pattern (arm approaches plausibly, grasp fails).
Secondary hypothesis: `control_freq=10` not matching the demos' true native
collection frequency (see the `control_freq` config comment).

### Instrumentation added to investigate this (not yet run)

Per request, added actual instrumented-rollout capability rather than
guessing further from summary statistics alone:

- `logging_schema.py`: `state_chunk` (per-step `[eef_pos_x,y,z,
  gripper_qpos_0,1]`, from the raw env observation -- not the policy-facing
  8D `observation.state`, which also needs quat→axis-angle and isn't needed
  for this check) and `video_frame_steps` (which steps got a saved frame).
- `replay.py`: `replay_episode()` now accepts `video_dir` /
  `video_sample_every_n_steps` -- every Nth step, `env.render()` is called
  and saved as `step_<i>.png`. Off by default (extra render() calls).
- `plots.py`: `plot_episode_trajectory()` — eef position (x,y,z), gripper
  qpos, and `action[:,6]` over time for one episode, with an optional
  second (`compare_*`) series overlaid for later policy-vs-demo comparison.
- `config.py` / `configs/demo_replay_smoke.yaml`: `video_sample_every_n_steps:
  20` (on for the smoke config -- this is exactly the run that needs looking
  at) and `save_trajectory_plots: true` (cheap, on everywhere). Both local
  disk only (`outputs/*/videos/`), not synced to Drive -- meant for
  interactive inspection within the session (file browser / downloading a
  few PNGs), not long-term storage.

All new logging/plotting/frame-saving code is covered by local tests against
`MockLiberoEnv` (`tests/test_wiring_and_metrics.py`) since it only needed
`env.step()`/`env.render()` and matplotlib, no simulator. What's NOT yet
verified: whether real `LiberoEnv.render()` frames actually show the arm
relative to the object clearly enough to be useful, and whether `state_chunk`
correctly reflects real (not mock) `robot_state` values. That needs an actual
Colab run of the updated smoke config.

## Design constraints encoded in this codebase (do not violate)

- `n_decision_points` (config field `n_decision_points`, default `1`) must be applied identically across Oracle / World-model / Random conditions — enforced in code, not by convention.
- `exec_horizon` (how many env steps run per policy call) and `selection_unit` (granularity at which candidates are compared) are separate config fields even before `selection_unit` has more than one implemented value. Step-level selection would break chunk-internal structure and produce a fake performance drop — this is a documented invariant, not a hypothetical.
- No monkey-patching `lerobot` or `libero`. If ever unavoidable, isolate in its own module with a comment explaining why.
- All experiment parameters live in `configs/*.yaml`. No hardcoded constants in `xgap_code/` or `scripts/`.
