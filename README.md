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

### Instrumentation added to investigate this (not yet run against a real sim)

Per request, added actual instrumented-rollout capability rather than
guessing further from summary statistics alone:

- `logging_schema.py`: `state_chunk` (per-step `[eef_pos_x,y,z,
  gripper_qpos_0,1]`, from the raw env observation -- not the policy-facing
  8D `observation.state`, which also needs quat→axis-angle and isn't needed
  for this check) and `video_frame_steps` (which steps got a sampled frame).
- `replay.py`: `replay_episode()` now accepts `video_path` /
  `video_sample_every_n_steps` -- every Nth step, `env.render()` is called;
  all sampled frames for the episode are written as **one `.mp4`** at
  `video_path` (via `imageio[ffmpeg]`, already in `requirements.txt`), not a
  folder of PNGs (an earlier version did that -- one video file per episode
  is easier to actually watch, which was the point). Playback fps defaults to
  `control_freq / video_sample_every_n_steps`, i.e. roughly real-time. Off by
  default (extra render() calls).
- `plots.py`: `plot_episode_trajectory()` — eef position (x,y,z), gripper
  qpos, and `action[:,6]` over time for one episode, with an optional
  second (`compare_*`) series overlaid for later policy-vs-demo comparison.
- `config.py` / `configs/demo_replay_smoke.yaml`: `video_sample_every_n_steps:
  1` (every step, on for the smoke config -- this is exactly the run that
  needs watching; ~300-step episodes at 10Hz -> ~30s videos) and
  `save_trajectory_plots: true` (cheap, on everywhere). Both local disk only
  (`outputs/*/videos/`), not synced to Drive -- meant for interactive
  inspection within the session (Colab's file browser plays `.mp4` inline),
  not long-term storage.

All new logging/plotting/video-writing code is covered by local tests
against `MockLiberoEnv` (`tests/test_wiring_and_metrics.py`, 13 tests) --
including the actual `imageio[ffmpeg]` mp4-encoding path, installed and
verified locally this session specifically because writing video is exactly
the kind of thing that silently degrades to a different codec/plugin when a
dependency is missing (it did, once, locally: without `imageio-ffmpeg`
installed, `imageio` silently fell back to a TIFF writer and crashed on the
`fps` kwarg -- caught here, not in Colab). What's still NOT verified locally
(needs lerobot/libero, Linux-only): whether real `LiberoEnv.render()` frames
actually show the arm relative to the object clearly enough to be useful, and
whether `state_chunk` correctly reflects real (not mock) `robot_state`
values. That needs an actual Colab run of the updated smoke config.

## Visual diagnosis of the 0/10 result

The smoke config's instrumentation (above) was run for real, and it worked:
real `LiberoEnv.render()` frames and `state_chunk` both came back usable.

**What the videos show.** Watching `demo_replay_relative__..._0.mp4`: the arm
reaches for and grasps a real object, carries it off-screen, comes back
empty-handed, then near the basket closes the gripper again -- this time
fully, on nothing. The trajectory plot for this same episode confirms it
numerically: `gripper_qpos_0` drops only to ~0.027 during the first close
(fingers stopped by something between them -- a real grasp) but drops all the
way to ~0.0 during the second close (nothing between the fingers -- closed on
air). `action[:,6]` shows the matching open→close→open→close→open command
sequence. Task `libero_10:0` is a two-object task ("put both the alphabet
soup and the tomato sauce in the basket"), so two grasp attempts is expected
-- the first one works, the second one doesn't find its target.

**Bonus finding from the same plots:** `relative` mode's eef trajectory is
smooth and stays in a bounded, physically-plausible range; `absolute` mode's
is visibly jittery/erratic (eef_z spikes past 1.0 partway through). This
matches what you'd expect if the raw demo actions are genuinely small deltas
-- interpreting the same small numbers as absolute world-frame targets every
step produces jumpy, inconsistent motion. `success=False` in both regardless,
but this is independent evidence `relative` is the physically sane
interpretation, beyond what the binary success-rate comparison alone shows.

**A verification mistake, corrected before it became a false conclusion:** to
check whether the init state matches the demo, the dataset's own recorded
first frame was fetched directly (image bytes embedded in the data parquet,
same technique as the H3 baseline) and compared to the video's first frame.
The first attempt used the wrong reference -- it assumed `HuggingFaceVLA/libero`'s
flattened `tasks.parquet` row order (task index 0 = "put the white mug on the
left plate...") matches LIBERO's own internal `task_id` numbering for the
`libero_10` suite. **It doesn't.** Checked directly (`harness.get_libero_task_language("libero_10",
0)` in Colab, no simulation needed): LIBERO's own task 0 is "put both the
alphabet soup and the tomato sauce in the basket" -- `tasks.parquet` index 5,
not 0. (This is exactly the kind of ordering assumption `get_libero_task_language`
was written to avoid trusting -- the mistake here was in ad hoc manual
verification, not in `dataset_io.py`/`harness.py` themselves, which never
made this assumption.) Once corrected, the dataset's actual first frame for
the right episode (global `episode_index=8`, the first -- lowest-numbered --
episode matching this task, fetched directly from `data/chunk-000/file-003.parquet`)
**matches the video's first frame closely**: same basket position, same
objects (soup can, ketchup bottle, milk carton, juice box, two small boxes)
in the same relative layout. Camera zoom/crop differs slightly; object
placement does not.

**Net effect on the hypothesis ranking:** this weakens the init-state/`within_task_index`
hypothesis (the starting layout is right) and strengthens `control_freq`
(H1-adjacent) back to leading candidate -- a correct start followed by
plausible-but-imprecise motion that misses by the second, more demanding
manipulation is consistent with per-step actions being applied at the wrong
scale/duration, not with a wrong scene.

## `control_freq` was wrong from the start (confirmed from LIBERO's own source)

Before running the `control_freq=20` comparison, its premise was checked
directly rather than guessed: what rate were the *original* LIBERO
demonstrations actually collected/simulated at? `control_freq=10` (this
project's default up to this point) was chosen by trusting
`HuggingFaceVLA/libero`'s `meta/info.json` (`fps: 10.0`) as the recording
rate. That trust was misplaced.

**Confirmed from four independent places in LIBERO's own repository**
(`Lifelong-Robot-Learning/LIBERO`), not inferred:

- `libero/libero/envs/bddl_base_domain.py` -- `BDDLBaseDomain.__init__(...,
  control_freq=20, ...)`.
- `libero/libero/envs/env_wrapper.py` -- `OffScreenRenderEnv` (what
  lerobot's `LiberoEnv` wraps) defaults to `control_freq=20`.
- `scripts/collect_demonstration.py` -- the actual teleop collection
  environment is built with `control_freq=20`.
- `scripts/create_dataset.py` -- the script that produces LIBERO's
  **released `.hdf5` files** sets `control_freq=20`, writes it into every
  episode's own `env_args` attributes (self-documenting), and replays
  `actions` **1:1** (`for j, action in enumerate(actions): env.step(action)`),
  asserting state divergence `< 0.01` against the recorded states. This is
  direct proof: one stored action == one 20 Hz `env.step()`, no substeps
  folded in, no subsampling.

robosuite's own environment default (`robosuite/environments/robot_env.py`)
is also `control_freq=20` -- LIBERO doesn't override it, it just makes the
inherited default explicit.

**So where did `fps=10.0` come from?** Not from LIBERO. Traced to
OpenPI's conversion script (`examples/libero/convert_libero_data_to_lerobot.py`),
which **hardcodes `fps=10`** as a literal while iterating every recorded step
with no stride -- i.e. it mislabels a straight 1:1, unmodified 20 Hz action
sequence as "10 fps" in the output dataset's metadata. `HuggingFaceVLA/libero`,
`lerobot/libero`, and `physical-intelligence/libero` all report the identical
episode/frame counts (1693 episodes, 273465 frames), confirming they share
this same conversion lineage and the same mislabeled metadata field --
propagated forward, never actually verified against LIBERO's own source
until now.

**Fixed:** `control_freq` default changed `10 -> 20` in both
`configs/demo_replay_smoke.yaml` and `configs/demo_replay.yaml`. This also
happens to match what `lerobot`'s own `LiberoEnv`/`EnvConfig` already
defaulted to (`fps: int = 20`) -- the project's earlier override to `10`,
based on the mislabeled dataset field, was fighting lerobot's own correct
default the whole time.

### cf=10 vs cf=20: neither alone fully explains the failure, but together they localize it

Single-episode, single-variable comparison
(`configs/demo_replay_smoke_cf20.yaml` vs `configs/demo_replay_smoke.yaml`,
same task, same episode, `relative` mode), watched directly and then
compared numerically against the *actual recorded demo* (episode 8's own
`observation.state`/`action`, fetched straight from the dataset -- see below):

- **`control_freq=10`:** eef trajectory drifts increasingly from the
  recorded demo's own trajectory over the episode (arm ends up carried well
  outside the camera's framing by the end). The **first** grasp still
  succeeds -- `gripper_qpos` closes only partially (~0.027, matching the
  recorded demo's ~0.025 at the same point almost exactly), consistent with
  a real object between the fingers. The **second** grasp closes fully to
  ~0.0 -- nothing between the fingers -- and the episode ends with the arm
  well off from where the recorded demo ends.
- **`control_freq=20`:** overlaying our replay's logged `state_chunk` /
  `action_chunk` directly against the recorded demo's own `observation.state`
  / `action` for the same episode (`plot_episode_trajectory(...,
  compare_state_chunk=..., compare_action_chunk=...)` -- see
  `outputs/demo_replay_smoke_cf20/_local/videos/`) shows the eef position
  trajectory (x, y, z) tracking the recorded demo **almost exactly** through
  the first grasp phase, with only a small, slowly-growing gap afterward --
  a dramatically better positional match than `control_freq=10`. And yet:
  **neither grasp succeeds** -- `gripper_qpos` closes fully to ~0.0 both
  times, where the recorded demo shows a partial close (~0.025) both times.
  `action[:,6]` (the commanded gripper signal) is pixel-identical to the
  recorded demo throughout -- the command sequence itself was never in
  question, only what happens physically when it's executed.

**This is the key new fact: `control_freq=20` gets eef position essentially
right and still fails to grasp.** That decouples two previously-conflated
questions -- "does the arm get to the right place" (yes, at cf=20) and "does
a correctly-positioned gripper actually close on the object" (no, at either
`control_freq`). The comparison plot only tracks position + `gripper_qpos`
(`state_chunk`'s 5 dims -- see `replay.py`'s `_extract_state`), not
**orientation** (the axis-angle/quaternion component of the real 8D
`observation.state`, deliberately left out of `state_chunk` early on since
it wasn't needed for the position-only checks up to this point). A gripper
that arrives at the exactly correct x/y/z but a few degrees off in approach
angle would grasp air exactly like this, at any `control_freq`.

**Leading hypothesis now: orientation.** Not yet tested -- would need
`state_chunk` (or a parallel field) extended to include `eef_quat`, plotted
against the recorded demo's own orientation the same way position already
is. Session paused here (mid-investigation) to consolidate findings before
continuing.

### Methodology note worth keeping: compare against the recorded demo directly, not just visually

The `compare_state_chunk`/`compare_action_chunk` overlay technique used above
-- fetch the *actual* recorded episode's `action`/`observation.state`
directly from the dataset (same raw-parquet-column-read technique as the H3
baseline, no `LeRobotDataset` needed for this) and plot it against our own
replay's logged trajectory with `plot_episode_trajectory` -- turned out to be
far more diagnostic than watching video alone (it caught the
first-grasp-succeeds-second-doesn't asymmetry, and the
position-matches-but-grasp-still-fails split, neither of which was obvious
from video by itself). Worth reusing this pattern in step 3 for policy-vs-demo
comparisons once there's a policy rollout to compare.

## Two cheaper, more decisive tests than continuing the orientation guess

Before extending `state_chunk` to orientation, two tests were proposed that
each bisect the remaining hypothesis space more directly, at lower cost.
Both added to `notebooks/run_replay.ipynb`; neither run yet.

**1. Sanity check against a KNOWN answer.** Run `lerobot`'s own
`lerobot-eval` CLI against `lerobot/pi05_libero_finetuned` -- a checkpoint
with a *published* number (96% on LIBERO-10, per lerobot's own
`docs/source/libero.mdx`, "Reproducing published results"). This touches
**zero lines of this project's code** and needs no 50GB dataset download
(eval only needs the environment + the small policy checkpoint). If it
reproduces ~96%, that confirms installation / MuJoCo / success-checking /
init-state-selection are all fine at the infrastructure level, and narrows
any remaining bug to this project's own harness specifically (init-state
indexing or the action-injection path). If it comes back ~0%, the
environment layer itself is broken independent of anything in `xgap_code/`
-- no amount of harness fixing would show up as a result until that's
addressed first. Twelve Colab round-trips so far have all stayed inside our
own code; this is the first test of that premise itself.

**2. Sweep every candidate `init_state` against one fixed demo.**
`scripts/sweep_init_states.py` / `configs/sweep_init_states.yaml`: replay
the SAME recorded action sequence (libero_10 task 0, dataset episode_index=8
-- the same episode used throughout the comparisons above) against every
`init_state` LIBERO has for this task (`harness.get_num_init_states()`, the
exact function `LiberoEnv` itself calls internally -- not a guessed count).
No new instrumentation, one loop, reuses `EpisodeStore` for resume/logging.
If any index succeeds, `within_task_index` (`dataset_io.py`) is confirmed
picking the wrong LIBERO init_state, and the successful index tells us the
right one directly. If none succeed across the full set, init-state
indexing is exonerated outright and orientation becomes the next thing to
actually add and check, not just suspect.

## Sanity check result: infrastructure is confirmed sound, the bug is ours

Ran the `lerobot-eval` sanity check for real. Two things worth recording,
both caught from an actual run, neither predicted:

**`--env.task=libero_10` evaluates all 10 tasks in the suite, not "5
episodes."** `--eval.n_episodes=5` is *per task*, and `--env.task_ids` was
not set to restrict it -- so this ran 5 episodes × 10 tasks = up to 50
episodes, not the 5 the notebook cell's own description implied. Worth
fixing the cell (`--env.task_ids=[0]` to actually get a quick 5-episode
check) before running this again.

**The `running_success_rate=0.0%` that caused real alarm mid-run was a
false signal, twice over:**

1. `Stepping through eval batches: N/5` is a **per-task** progress bar --
   every time it finishes one task's 5 episodes and starts the next, the
   displayed `running_success_rate` resets to reflect zero completed
   episodes *of the new task*, not zero overall. Reading "0.0%" at 25%/50%
   of a bar that had just reset looked like an ongoing failure; it was a
   fresh counter.
2. Piping `lerobot-eval`'s tqdm output to a file (the log-file redirect
   fix, above) produces a new line per step update rather than overwriting
   in place -- a 1.3MB, 10709-line log from under two full tasks' worth of
   steps. Skimming the tail alone (or even the last ~200 lines) can land
   inside an in-progress episode's step-by-step climb from 0 to 520, which
   *always* shows 0.0% until that episode finishes -- structurally, not as
   a finding. Grepping the whole file for `100%` (the per-task completion
   marker) was what actually resolved it.

**Actual result, from the completed tasks:** every task that finished ran
5/5 episodes at **`running_success_rate=100.0%`** (confirmed across 4
completed tasks before the run was stopped, i.e. 20/20 episodes). This
answers the sanity check decisively: **installation, MuJoCo, rendering,
success-checking, and init-state selection are all sound at the
infrastructure level.** Twelve-plus round-trips of assuming the bug must be
somewhere in this project's own harness were operating on the right
premise -- this is the first time that premise was actually tested rather
than assumed, and it holds. The bug is confined to `xgap_code` (most likely
`within_task_index` or the action-injection path) or to something specific
about how the SmolVLA checkpoint under test differs from
`pi05_libero_finetuned` -- not the shared environment layer underneath both.

Next: the init-state sweep (`scripts/sweep_init_states.py`) -- results below.

## Init-state sweep result: indexing bug confirmed, general mapping still unknown

Ran `scripts/sweep_init_states.py` for real: libero_10 task 0
(`demo_within_task_index=0`, dataset `episode_index=8`), the SAME demo used
in every comparison above, replayed against all 50 candidate init_states
LIBERO has for this task.

**Only init_state 3 and 23 (of 50) succeed** -- both required objects placed
in the basket -- when replaying this demo's own recorded actions.
`run_demo_replay.py`'s current caller convention
(`demo_episode_index_within_task=episode_seed` in `harness.make_real_libero_env`)
had been running this exact demo against **init_state 0**, which is not one
of the two that work. This directly confirms the leading suspect from the
"both low" branch of `decide_env_convention`: the bug is `within_task_index`
(`dataset_io.py`) being reused as a LIBERO init_state index without actually
being one.

**False alarm along the way, resolved by reading source, not guessing:** the
video for index 3 appeared to show a milk carton, raising a "wrong task got
loaded" concern. Checked task 0's actual BDDL file directly
(`suite.get_task_bddl_file_path(0)`, LIBERO's own path resolver, not a
reconstructed path) -- its object list is `alphabet_soup, cream_cheese,
tomato_sauce, ketchup, orange_juice, milk, butter, basket`. `milk` is a real
distractor object genuinely in this scene; task 0 is confirmed unchanged
("put both the alphabet soup and the tomato sauce in the basket"). Several
of these grocery items (`milk`, `orange_juice`, `cream_cheese`, `butter`)
are all similarly-shaped cartons/boxes at 256x256 render resolution and are
easy to visually mix up with each other -- not a bug.

**Constant-offset hypothesis tested and rejected.** If `init_state =
within_task_index + 3` generalized, demo 1 (`demo_within_task_index=1`,
`configs/sweep_init_states_demo1.yaml`) should succeed at init_state 4 and/or
24 (the same two candidates, shifted by 1). Checked with
`scripts/render_init_state_video.py` (a standalone, always-re-runs script
added specifically because `sweep_init_states.py`'s resume logic would
otherwise silently skip re-running an already-completed index even after
turning video on for it). Both **partially** succeeded -- the first of the
two required objects was placed at both indices -- but neither reached full
task success; the second object's placement failed both times. The simple
constant-offset theory does not hold.

**Checked whether the dataset stores the init_state directly** (would make
all of the above unnecessary): inspected the full parquet schema of a shard
file. Only standard LeRobot columns are present
(`observation.images.image[2]`, `observation.state`, `action`, `timestamp`,
`frame_index`, `episode_index`, `index`, `task_index`) -- no init_state or
seed column. The true mapping is not recoverable from dataset metadata; it
would need to come from the original LIBERO `.hdf5` -> LeRobot conversion
source, or be found empirically per demo (a full 50-way sweep per demo,
expensive).

**Verdict for step 2:** `env` / success-detection / init-state mechanism are
all confirmed sound -- LIBERO's own init-state selection genuinely
reproduces full task success when recorded actions are replayed against the
init_state they were actually collected from. The bug behind the originally
reported symptom (0% success, arm approaches plausibly, never completes the
grasp) is `xgap`'s own `within_task_index` -> LIBERO init_state mapping, not
the environment, not LIBERO's success-checking, and -- per the earlier
`lerobot-eval` sanity check -- not the shared infrastructure layer either.

Finding the *general* mapping rule (so a full replay run uses every demo's
correct init_state) is fix-stage work and out of scope for this diagnostic;
recorded here as the concrete next action once fixing begins, not pursued
further now.

One loose thread worth flagging for that future work: in both *partial*
successes above (init_state 4 and 24, demo 1), the approach to the second
object visually tracked a plausible trajectory right up to the grasp -- not
a trajectory that missed the area entirely. This is the same qualitative
shape as the project's original reported symptom, observed here under an
init_state already known NOT to be this demo's true match -- consistent
with (not independent confirmation of) the coarse position-mismatch
explanation, and does not on its own revive the orientation hypothesis.

## Mapping search abandoned: wrong question, and demo replay already did its job

Stopped chasing the general `within_task_index -> init_state` rule. Two
things wrong with continuing it:

1. **Init-state indices only look like a "match" -- they aren't necessarily
   one.** Demo 0 succeeding at 2 of 50 candidates says those two are *close
   enough* to reproduce success with this demo's open-loop actions, not that
   either is the demo's actual recorded initial state. LIBERO's shipped
   `init_states` array is a separate, fixed evaluation set -- there is no
   guarantee the demo's true initial state is a member of it at all. A
   "general mapping rule" between dataset episode order and this array may
   simply not exist to be found.
2. **Policy rollouts never use a demo's initial state in the first place.**
   Standard LIBERO/lerobot evaluation steps through `init_states` in plain
   order starting at index 0 -- which is exactly what the `lerobot-eval`
   sanity check ran, and it hit 100%. The `within_task_index` mapping is a
   concern specific to *demo replay* (this project's own diagnostic
   harness), not to how SmolVLA (or any policy) will actually be evaluated.

Demo replay's actual purpose -- checking whether env / success-detection /
init-state selection are trustworthy -- is already answered: they are (see
the sweep result above, and the `lerobot-eval` sanity check before it).
Finding the exact demo-to-init_state correspondence would only matter if
demo replay itself needs to run again at scale, which is not the current
need.

**If demo replay ever needs precise, guaranteed-correct init states again**,
the reliable path is bypassing LIBERO's `init_states` array entirely:
LIBERO's original `.hdf5` demo files store the actual per-timestep simulator
state under each demo's own `states` dataset (`demo_X/states`); `states[0]`
is the exact flattened MuJoCo state the demo was recorded from.
Robosuite/LIBERO's own demo-playback tooling restores this via
`env.sim.set_state_from_flattened(state)` followed by `env.sim.forward()`,
rather than resetting through `init_states[idx % len(init_states)]` at all.
This sidesteps the whole indexing question -- no mapping to find, no
"close enough" ambiguity -- at the cost of needing the original `.hdf5`
files (not just the LeRobot-converted parquet dataset) and one small,
isolated, documented use of the underlying `env.sim` (not `libero`/`lerobot`
monkey-patching -- calling a public robosuite method they don't wrap).
**Noted here for later, not implemented now.**

## Next: SmolVLA directly via `lerobot-eval`, no xgap code

Bypassing this project's own harness entirely for the actual checkpoint
under test (`HuggingFaceVLA/smolvla_libero`), same as the `pi05` sanity
check above -- this is the more direct answer key `xgap`'s own N=1 result
needed all along, and doubles as the Gate-1 baseline number.

```
!lerobot-eval \
    --policy.path=HuggingFaceVLA/smolvla_libero \
    --env.type=libero \
    --env.task=libero_10 \
    --env.task_ids=[0] \
    --eval.batch_size=1 \
    --eval.n_episodes=5 \
    --env.max_parallel_tasks=1 \
    > /content/lerobot_eval_smolvla.log 2>&1
!tail -n 80 /content/lerobot_eval_smolvla.log
```

`--eval.n_episodes` is per-task (see the false alarm above) -- with
`--env.task_ids=[0]` this is 5 episodes total, not 50. No
`--policy.n_action_steps` override here on purpose: let the checkpoint's own
config supply it rather than reusing `pi05`'s `10` by copy-paste.

- **A usable, nonzero number** -> this is both the Gate-1 baseline and the
  answer key for `xgap`'s own N=1 result. The original harness-vs-policy
  question is answered without needing the mapping at all.
- **~0%** -> the checkpoint itself (not the shared environment, confirmed
  sound above) is the suspect. Next: `n_action_steps` sweep and wrist-camera
  input verification (H1, H2).

**Result: 0/5 (`pc_success=0.0`, `avg_sum_reward=0.0`), confirmed via pure
`lerobot-eval` with zero `xgap` code involved.** This is decisive: the
original N=1 symptom reproduces on completely independent infrastructure
already proven sound (the `pi05` sanity check above). The bug is not
`xgap`'s harness -- it is specific to this checkpoint (or this checkpoint's
interaction with the environment). Also notable: `eval_ep_s=462s` (~7.7 min
per episode) at the checkpoint's default `n_action_steps=1` -- a fresh
policy call every single env step, closed-loop, using only the first of 50
predicted actions each time. Worth keeping in mind for time-budgeting H1
below (higher `n_action_steps` means fewer policy calls per episode, likely
faster as well as a different execution granularity).

Wrist-camera input (H2) is *not* a live suspect for this specific 0%
result -- `lerobot-eval`'s own `LiberoEnv` supplies both `image` and
`image2` automatically (that's the point of bypassing `xgap`'s harness
entirely here); H2 only mattered as a suspect for `xgap`'s *own* harness
potentially dropping a camera, which this run doesn't go through at all.
So H1 (`n_action_steps`) is the next -- and only remaining -- item from the
original hypothesis order to test against this checkpoint via `lerobot-eval`.

### H1: `n_action_steps` sweep, still via pure `lerobot-eval`

Same command as above, sweeping `--policy.n_action_steps` one value at a
time (sequential, per the original H1->H2->H4 order -- not exhaustive).
`1` (the checkpoint's own default) already ran: 0/5. Next candidates, in
order: `10` (the value that worked for `pi05`'s published reproduction --
not assumed correct for this checkpoint, just the next thing to try),
then `25`, then the full `50` (`chunk_size`, fully open-loop) if `10` also
fails.

```
!lerobot-eval \
    --policy.path=HuggingFaceVLA/smolvla_libero \
    --policy.n_action_steps=10 \
    --env.type=libero \
    --env.task=libero_10 \
    --env.task_ids=[0] \
    --eval.batch_size=1 \
    --eval.n_episodes=5 \
    --env.max_parallel_tasks=1 \
    > /content/lerobot_eval_smolvla_nas10.log 2>&1
!tail -n 80 /content/lerobot_eval_smolvla_nas10.log
```

**Result: 0/5 again (`pc_success=0.0`).** H1 (execution granularity) is
rejected -- 1 vs 10 makes no difference to success. `eval_ep_s` dropped from
462s to 145.6s (~3.2x), on both a GPU upgrade (T4 -> A100) *and* a 10x
reduction in policy calls per episode at the same time -- far less than the
~10-30x a compute- or call-count-bound workload would show. Side finding:
the real time bottleneck here is very likely env stepping/rendering, not
the policy forward pass -- consistent with SmolVLA being a small model.
Not pursued further (off the critical path for the actual diagnosis), but
worth remembering if runtime budget becomes an issue later.

Continuing the `n_action_steps` axis to `25`/`50` next would only ever
produce a third "0/5, learned nothing new" data point -- there are two
higher-information, still-untested axes first, reordered ahead of it below.
`25`/`50` are kept as the fallback if both come back inconclusive.

### Higher-priority than finishing the `n_action_steps` sweep

**Does this checkpoint know `libero_10` at all?** The model card doesn't
state which LIBERO suite(s) it was trained on -- genuinely unknown, not
assumed. Running `libero_spatial` (a different suite, same
`n_action_steps=10`) splits two very different explanations that a 0% on
`libero_10` alone cannot: success here would mean "the checkpoint/policy
loop works, `libero_10` specifically is the problem" (wrong config/data
for this suite); another 0% narrows it to the checkpoint itself, suite
notwithstanding. More decisive than another point on the `n_action_steps`
axis either way.

```
!lerobot-eval \
    --policy.path=HuggingFaceVLA/smolvla_libero \
    --policy.n_action_steps=10 \
    --env.type=libero \
    --env.task=libero_spatial \
    --env.task_ids=[0] \
    --eval.batch_size=1 \
    --eval.n_episodes=5 \
    --env.max_parallel_tasks=1 \
    > /content/lerobot_eval_smolvla_spatial_nas10.log 2>&1
!tail -n 80 /content/lerobot_eval_smolvla_spatial_nas10.log
```

**Result: 3/5 (`pc_success=60.0`), episodes 0/2/4 succeeded, 1/3 failed.**
Decisive: the checkpoint/policy/env/`lerobot-eval` loop all genuinely work
-- this rules out "the checkpoint itself is broken" outright. `libero_10`
specifically is where this checkpoint fails, not SmolVLA in general. This
also weakens the resolution hypothesis below somewhat before even running
it: `libero_spatial` used the exact same default (360) render resolution
as every `libero_10` run above, and still succeeded 60% of the time -- if
a 360-vs-256 mismatch were catastrophic, it should tank every suite
roughly equally, not just this one. Still worth running (cheap, already
queued, and `libero_10`'s specific scenes could still interact with it
differently), just no longer the leading theory.

Remaining candidate explanations for the `libero_10`-specific 0%: (a)
genuine task difficulty -- `libero_10` (LIBERO-Long) tasks are two-stage
("put BOTH X and Y in the basket"), materially harder than `libero_spatial`'s
single-stage tasks, and the checkpoint may simply be weak on long-horizon
composition rather than buggy; (b) something specific to `libero_10`'s
scenes/objects (camera framing, clutter, the multi-object BDDL layout) that
doesn't show up in `libero_spatial`. Neither is confirmed yet -- flagging
both rather than picking one without evidence.

**Resolution mismatch, unresolved since Step 1.** The env's default render
resolution is 360, the checkpoint declares 256 as its input, and there's a
separate 512-padding setting on top of that (see the checkpoint config
table near the top of this file) -- where (or whether) a resize actually
reconciles these was never confirmed from source, only left as an open
question. If it doesn't, the policy is seeing images shaped differently
than it was trained on. Cheap to test directly: force the env to render at
256 instead of relying on an implicit resize.

First confirm the actual CLI field names before spending eval time on a
typo'd flag (`harness.py`'s `make_real_libero_env` uses
`observation_height`/`observation_width` as constructor kwargs -- check
these are the same names `lerobot-eval`'s `--env.*` CLI exposes):

```python
import dataclasses
from lerobot.envs.configs import LiberoEnv
print([f.name for f in dataclasses.fields(LiberoEnv)])
```

**Confirmed:** `observation_height`/`observation_width` are real
`LiberoEnv` fields (full list: `task`, `fps`, `features`, `features_map`,
`max_parallel_tasks`, `disable_env_checker`, `task_ids`, `episode_length`,
`obs_type`, `render_mode`, `camera_name`, `init_states`,
`camera_name_mapping`, `observation_height`, `observation_width`,
`is_libero_plus`, `control_mode`) -- the eval command below uses the right
flag names.

Then, same baseline (`libero_10`, `n_action_steps=10`) with only resolution
changed:

```
!lerobot-eval \
    --policy.path=HuggingFaceVLA/smolvla_libero \
    --policy.n_action_steps=10 \
    --env.type=libero \
    --env.task=libero_10 \
    --env.task_ids=[0] \
    --env.observation_height=256 \
    --env.observation_width=256 \
    --eval.batch_size=1 \
    --eval.n_episodes=5 \
    --env.max_parallel_tasks=1 \
    > /content/lerobot_eval_smolvla_res256.log 2>&1
!tail -n 80 /content/lerobot_eval_smolvla_res256.log
```

**Result: 1/5 (`pc_success=20.0`), episode 3 only.** Weak, inconclusive
signal at this sample size -- 0/5 vs 1/5 is a single episode, well within
noise. Not a clean "confirmed" or "rejected" the way `libero_spatial` was.
Not pursued further with more episodes for now (diminishing value relative
to cost); noted as a real but unresolved data point, not closed.

**Alongside success rate, log gripper-close duration for every run above,
not just pass/fail.** Three low-or-zero results in a row say little on
their own, but whether the longest continuous closed-gripper run (see
`xgap_code/gripper_metrics.longest_close_run`, and
`DEMO_MIN_ACTUATION_LAG_STEPS=15` as the "a real demo grasp holds the
gripper shut for 15-20+ consecutive steps" reference point measured
earlier from real demo data) moves between conditions distinguishes "never
attempts to close" from "closes but doesn't actually grasp" -- two
completely different bugs.

Checked whether pure `lerobot-eval` already saves raw per-step actions:
`eval_info.json` (2.5KB) turned out to be exactly the same aggregated
summary already seen in stdout (`per_task`/`per_group`/`overall`, no raw
actions) -- confirmed by reading it directly, not assumed from size alone.
The module-path guess (`lerobot.scripts.eval`) was also wrong; the real
one, found via `cat $(which lerobot-eval)`, is `lerobot.scripts.lerobot_eval`.

**Reading that module's actual source paid off: no new script needed.**
`eval_policy()` (and the `rollout()` it calls) already has a full
recording path -- writes a real `LeRobotDataset` (raw `action` per frame,
same on-disk shard layout as `HuggingFaceVLA/libero`) to
`<output_dir>/recordings/<task_group>_<task_id>/` -- but `eval_main()`
only wires it up when `cfg.eval.recording` is `True`; the CLI never sets
it by default, which is why it never appeared. Confirmed the flag names
directly from source (`cfg.eval.recording`, `cfg.eval.recording_repo_id`,
`cfg.eval.recording_private` are literally referenced in `eval_main()`),
not guessed. One caveat read from the same source: turning `recording` on
sets `videos_dir = None` and `max_episodes_rendered = 0` -- the standalone
`.mp4` files stop being written for that run (images are still captured,
just packaged as the recorded dataset's own video feature instead of loose
files).

Added `scripts/gripper_close_from_recording.py` -- reads
`<recording_dir>/data/chunk-*/file-*.parquet` (local files, same
column-projection technique `dataset_io.py` already uses for real demo
shards -- just pointed at a local path instead of `hf_hub_download`) and
reports `longest_close_run` per episode via the existing, already-tested
`gripper_metrics` function. No `recording_repo_id` passed, so nothing gets
pushed to the Hub -- stays local only.

```
!lerobot-eval \
    --policy.path=HuggingFaceVLA/smolvla_libero \
    --policy.n_action_steps=10 \
    --env.type=libero \
    --env.task=libero_10 \
    --env.task_ids=[0] \
    --eval.batch_size=1 \
    --eval.n_episodes=5 \
    --eval.recording=true \
    --env.max_parallel_tasks=1 \
    --output_dir=/content/outputs/eval_gripper_libero10 \
    > /content/lerobot_eval_smolvla_gripper_libero10.log 2>&1
!tail -n 40 /content/lerobot_eval_smolvla_gripper_libero10.log
!python {XGAP_DRIVE_ROOT}/scripts/gripper_close_from_recording.py \
    --recording-dir /content/outputs/eval_gripper_libero10/recordings/libero_10_0
```

(`--output_dir` pinned explicitly here, rather than relying on the
auto-generated timestamp directory used by every run above, so the
recordings path is known ahead of time instead of needing to be read back
out of the log.)

**Not run** -- watching the plain videos from the earlier `n_action_steps=10`
baseline run answered the question directly, no parquet/recording pipeline
needed. (Those specific video files were lost to a runtime restart before
they could be watched -- `lerobot-eval`'s `--output_dir` defaults to a path
under `/content/`, ephemeral VM disk, with no Drive-sync of its own the way
`xgap`'s own `EpisodeStore` has. Re-running the same baseline with
`--output_dir` pointed at a Drive path fixes this for next time:
`--output_dir=/content/drive/MyDrive/xgap/outputs/<name>`.)

## Verdict: gripper oscillates, doesn't just miss or never try

Watched the `libero_10` baseline videos (`n_action_steps=10`, default
resolution, 0/5). The gripper is **not** stuck open the whole episode, and
it's **not** one clean close-then-miss either -- it repeatedly closes and
re-opens while approaching the object, as if repeatedly attempting and
aborting the grasp. Qualitatively distinct from both halves of the
"never attempts to close" vs. "closes but doesn't grasp" question this was
meant to answer -- it's closer to indecisiveness than either a dead
gripper or a single failed attempt. (The quantitative
`longest_close_run` check, via `scripts/gripper_close_from_recording.py`
above, would confirm this numerically -- no single continuous close run
should reach the demo's 15-20+ step grasp threshold -- but wasn't run;
the visual signal alone was decisive enough to stop here.)

**Putting the whole SmolVLA-direct investigation together:**

- Checkpoint/policy/env/`lerobot-eval` loop are all confirmed working
  (`libero_spatial`: 60%, 3/5).
- The failure is specific to `libero_10`, not SmolVLA in general.
- `n_action_steps` (1 vs 10) makes no difference -- not an execution-
  granularity bug.
- Forcing render resolution to 256 gave a weak, inconclusive signal
  (0/5 -> 1/5) -- not ruled out, but not the leading explanation either,
  especially since `libero_spatial` already succeeded at the same default
  360 resolution.
- The gripper visibly *tries*, repeatedly, rather than never engaging or
  failing once cleanly -- an oscillating, low-confidence grasp behavior
  specifically on `libero_10`.

**Most consistent explanation: this is a checkpoint weakness on
`libero_10` specifically, not an environment or `xgap` harness bug.**
`libero_10` (LIBERO-Long) tasks are two-stage ("put BOTH X and Y in the
basket") and structurally harder than `libero_spatial`'s single-stage
tasks; the oscillating grasp behavior is a plausible signature of a policy
that is uncertain in exactly this harder regime, rather than of a
plumbing bug -- every plumbing-shaped explanation tested (resolution,
execution granularity, wrist camera, the environment/success-detection
layer itself, the demo-replay init-state mechanism) came back negative or
inconclusive-and-secondary. This also reproduces the project's original
reported symptom ("arm approaches objects plausibly, never completes the
grasp") exactly -- on completely independent infrastructure from the
`xgap` harness that symptom was first observed through.

Not confirmed, and out of scope for this diagnostic: *why* `libero_10`
specifically is hard for this checkpoint (undertrained on long-horizon
composition vs. something about `libero_10`'s scenes) -- that's a
model-training/fine-tuning question, not an environment-diagnosis one.

## Open thread that could overturn the verdict above: is the policy's actual input mirrored?

Raised after the verdict above was already written: object text ("Milk",
"Orange Juice") in the `libero_10` video frames reads as a mirror image.
If the tensor actually fed to `policy.select_action()` is left-right
flipped relative to what the checkpoint was trained on, that alone would
explain the observed grasp failures -- and would mean the "checkpoint
weakness" verdict above is wrong.

**Critical distinction: the saved `.mp4` path and the policy-input path are
different code paths.** The video comes from `env.render()` inside
`eval_policy()`'s `render_callback`; the policy sees whatever
`preprocess_observation() -> env_preprocessor() -> preprocessor()`
produces from `env.step()`/`env.reset()`'s own returned observation. A flip
in one does not imply a flip in the other -- must be checked directly,
not inferred from the video.

**Important prior constraint on the answer:** `pi05_libero_finetuned` hit
20/20 on this exact `lerobot-eval` CLI, this exact `libero_10` env config.
If policy input were mirrored, `pi05` should have failed too -- it didn't.
So the leading expectation is that the mirroring is confined to the video
save path (`env.render()`'s own convention, unrelated to what the policy
receives), not the policy input. But this is an expectation, not a
result -- confirming it either way is required; if the policy input turns
out to be mirrored too, the verdict above does not hold and needs to be
revisited (possibly explained by `pi05` being more flip-robust, or
something else not yet considered).

Added `scripts/check_image_mirroring.py`: builds the observation the exact
same way `lerobot-eval`'s own `rollout()` does internally (same public
functions -- `make_env`, `make_policy`, `make_pre_post_processors`,
`make_env_pre_post_processors`, `preprocess_observation` -- called
directly in the same order, verified from `lerobot/scripts/lerobot_eval.py`
source rather than reimplemented from guesswork; no monkey-patching), for
one `env.reset()` with no stepping, and dumps the resulting per-camera (
`image`=agentview, `image2`=wrist) tensors as PNG -- this is what the
policy actually receives, not a re-render for video. Also added
`dataset_io.fetch_reference_frame_images()` (same column-projection,
never-touch-the-whole-dataset pattern as the rest of that module) to pull
one frame of the same task/episode used throughout this project's
comparisons directly from `HuggingFaceVLA/libero`, as the ground-truth
orientation to compare against -- the standard here is "matches training
data's orientation", not "looks upright to a person".

Config construction uses `draccus.parse(EvalPipelineConfig, args=[...])`
directly (the same CLI args as the `n_action_steps=10` baseline run) --
unverified against source before use (unlike everything else in this
project), flagged here explicitly; if it errors, `lerobot.configs.parser`'s
actual source needs to be read to find the correct non-CLI entry point
instead of guessing further.

```
!python {XGAP_DRIVE_ROOT}/scripts/check_image_mirroring.py \
    --out-dir /content/drive/MyDrive/xgap/outputs/mirror_check
```

(Output pinned to Drive directly, not `/content/`, per the earlier lesson
about `lerobot-eval`'s own `--output_dir` not surviving a runtime reset.)

Four PNGs land in `outputs/mirror_check/`: `dataset_reference_observation_images_image.png`,
`dataset_reference_observation_images_image2.png`,
`policy_input_observation_images_image.png`,
`policy_input_observation_images_image2.png` -- compare each camera's
policy-input PNG against its matching dataset-reference PNG, side by side,
for a left-right flip.

**Result: not mirrored.** Both cameras' policy-input PNGs match their
dataset-reference PNGs' left-right orientation directly -- basket position,
gripper position, and the full left-to-right ordering of every grocery
item (cans, the `MILK`-labeled box, the orange juice carton) line up
between `dataset_reference_observation_images_image.png` /
`policy_input_observation_images_image.png`, and likewise for `image2`
(wrist). The mirrored text that prompted this whole check was specific to
the *saved `.mp4`* rendering path (`env.render()`), not what the policy
actually receives -- exactly the outcome the `pi05` 20/20 result predicted,
now directly confirmed rather than assumed. **This closes the one thread
that could have overturned the checkpoint-weakness verdict above -- it
stands.**

## Decision: the August gate moves to LIBERO-Spatial

LIBERO ships several task suites; this project has used two:
`libero_spatial` (single object, single pick-and-place) and `libero_10` /
LIBERO-Long (two objects, sequential -- e.g. "put both the alphabet soup
and the tomato sauce in the basket").

**Why the proposal originally picked LIBERO-Long: a ceiling-effect
argument.** This project needs to show performance degrading under
pressure -- impossible if the baseline is already near 100%. The proposal
argued easier suites like Spatial were saturated at SOTA (>98%), with
headroom concentrated in long-horizon tasks, and cited literature to
support it.

**The proposal's own criterion, applied to what was actually measured,
reverses the choice.** The proposal already acknowledged even Long
approaches saturation for top models (pi0.5 92.4%, TwinBrainVLA 95.4%,
VLA-GSE 96.8%) -- and judged "saturated" against the policy actually being
used, not the best model in the world, citing small-model numbers (OpenVLA
53.7%, Diffusion Policy 64.8%) as evidence there's still headroom for a
small policy. That's also why the proposal picked a small model
(SmolVLA) in the first place. Applying that same standard to what this
project actually measured:

| | LIBERO-Long | LIBERO-Spatial |
|---|---|---|
| top models | 92-97% | >98% |
| SmolVLA (measured this session) | 0% | 60% |
| usable as an experiment substrate? | no -- can't complete the task at all | yes |

SmolVLA's 60% on Spatial sits right inside the small-model headroom band
(53.7-64.8%) the proposal already used to justify small models having room
to move. It clears Gate 1 (would only fail if ≥90%) with room to spare, and
is not a ceiling -- there's space to observe degradation under pressure.
Long, by contrast, has no working baseline at all: SmolVLA cannot complete
the task, so there is nothing for `xgap`'s intervention conditions to be
measured against. This isn't overturning the proposal's criterion --
it's applying that exact criterion to real measured numbers, which the
proposal's own >98%-ceiling framing (a SOTA-relative claim) hadn't yet been
tested against.

**One acknowledged loss.** Part of the original case for Long was its
connection to WoVR's long-horizon error-accumulation framing, which the
planned H-step rollout design (`exec_horizon`, §6.1④) was meant to probe --
Spatial's shorter episodes leave less room to push `H` large. The (N, H)
2D sweep is an ICRA-scope extension, though; the August gate only needs
the N axis. If H-axis headroom becomes necessary later, Long is the
candidate to return to -- after additional fine-tuning of the checkpoint,
since a 0%-baseline suite still won't be usable as-is.

**How this gets written up:** "this checkpoint does not reliably complete
long-horizon two-stage tasks, so it was excluded from the August gate" --
more honest than forcing a measurement where there's no working baseline
to measure from. Long carries forward as an extension experiment, not a
dropped thread.

**Backup-suite discussion also flips.** The original backup candidate was
`libero_goal`/LIBERO-Plus, considered in case Long turned out too easy.
The real failure mode was the opposite (too hard). With Spatial as the
main suite, the backup question becomes "what if Spatial turns out too
easy" instead, with Long or `libero_goal` as candidates then.

## Harness parity check: does xgap's own env-construction path work?

Before scaling up to the real Gate-1 measurement (3-5 tasks × 15-20
episodes on `libero_spatial`), one thing needed checking first: does
`xgap`'s OWN harness (`harness.make_real_libero_env`) reproduce the same
real-world result as the official `lerobot-eval` CLI (which uses lerobot's
own `make_env`)? This matters because the actual Gate-1/N/H experiments
need control that `lerobot-eval`'s CLI doesn't expose --
`n_decision_points`/`exec_horizon`/`selection_unit`, the Oracle/World-model/
Random condition branching (see "Design constraints" below) -- so `xgap`'s
own harness has to be the execution path eventually, and its correctness
shouldn't be assumed just because the demo-replay half of it (env
construction, success detection) was already validated in a different
context (the init-state sweep, the sanity check).

**Live policy rollout is new -- everything before this point only replayed
pre-recorded demo actions.** Added `xgap_code/policy_rollout.py`
(`rollout_policy_episode` + `build_policy_and_processors`) and
`scripts/run_policy_rollout.py`. Policy loading and observation
preprocessing reuse lerobot's own public functions directly (`make_policy`,
`make_pre_post_processors`, `make_env_pre_post_processors`,
`preprocess_observation`) -- the same functions `lerobot-eval`'s own
`rollout()` calls internally, and the same construction pattern
`scripts/check_image_mirroring.py` already proved works end to end in this
exact Colab environment (that script's `_build_eval_cfg`-equivalent logic
is now factored into `build_policy_and_processors`). The env itself is
`harness.make_real_libero_env`, not lerobot's `make_env` -- that's the
whole point of this check.

**Init-state selection is STANDARD order, not the demo-matching remap.**
`episode_seed` (0, 1, 2, ...) is passed directly as
`demo_episode_index_within_task`, matching `lerobot-eval`'s own
`init_states[episode_index % len(init_states)]` convention. This is a
different use of that same harness parameter than demo replay's -- see
`harness.make_real_libero_env`'s updated docstring. No `dataset_io`, no
recorded demo actions, no init-state-matching question at all here; the
policy generates its own actions from live observations.

**One flagged, unverified risk.** lerobot's `preprocess_observation()` /
`env_preprocessor()` / `preprocessor()` are designed for lerobot's own
BATCHED `gym.vector.VectorEnv` (see `rollout()`'s type hint) -- but
`harness.make_real_libero_env` returns a single, UNBATCHED env (matching
`replay.py`'s existing usage throughout this project: plain `(7,)` actions,
unbatched obs dict). `policy_rollout.py` manually adds/removes a batch
dimension of size 1 around the shared preprocessing calls to bridge this.
Not yet verified against a real Colab run -- flagged in the module
docstring as the first place to look if this errors, rather than assumed
correct.

`configs/policy_rollout_libero_spatial_smoke.yaml`: `libero_spatial` task
0, 5 episodes, `HuggingFaceVLA/smolvla_libero`, one `policy.select_action()`
call per env step (matches the project's own confirmed n_action_steps=1-
vs-10 null result on `libero_10`, so a chunked executor isn't needed for
this parity check specifically). `max_steps=520` is inferred from progress-
bar output observed in earlier real `lerobot-eval` runs this session, not
confirmed from source -- watch `rollout_length` in this run's own output
against it.

```
!python {XGAP_DRIVE_ROOT}/scripts/run_policy_rollout.py \
    --config {XGAP_DRIVE_ROOT}/configs/policy_rollout_libero_spatial_smoke.yaml
```

- **Close to 3/5 (60%)** -> `xgap`'s harness is validated for policy
  rollout; proceed to the real Gate-1 measurement (3-5 tasks × 15-20
  episodes).
- **Meaningfully different** -> something in `xgap`'s env construction or
  the batching bridge above is wrong -- debug before trusting any number
  this harness produces, same discipline as every other step in this
  project.

Fixed one real bug caught by this run: `rollout_policy_episode` fed a
batched-but-not-`preprocess_observation`-shaped dict straight to
`env_preprocessor`, which errored `ObservationProcessorStep requires an
observation in the transition.` -- `preprocess_observation()` (the lerobot
function that shapes a raw env obs into the "transition" format its
processor pipeline expects) was missing from the call sequence, even
though `check_image_mirroring.py` already did this correctly.
`preprocess_observation` is now an injected parameter
(`preprocess_observation_fn`) rather than imported directly inside
`rollout_policy_episode`, so the function stays testable against
`MockLiberoEnv` with a plain identity-function stand-in -- a bare import
broke the local tests, which is what caught the missing call.

**Result: 2/5 (40%) via `xgap`'s own harness vs 3/5 (60%) via
`lerobot-eval` -- one episode apart, at n=5.** Not identical, but n=5 is
too small a sample to distinguish "harness bug" from ordinary noise (a
single episode flip moves this rate by 20pp either way), on top of likely
inherent run-to-run stochasticity in the checkpoint's own action sampling.
Both successes finished quickly (77, 73 steps, `longest_close_run` 36-39,
a plausible grasp shape); all three failures ran the full 520-step cap
rather than ending early -- an ordinary failure shape, not an obviously
broken one.

## Cost-aware sizing: comparison and measurement are the same run

Initially planned as two separate steps -- a harness-parity check at small
n, then a separate larger Gate-1 measurement. Collapsed into one: the
Gate-1 measurement (3-5 tasks × 15-20 episodes) already includes ~15-20
episodes on task 0, which IS the harness-parity check, at a much larger
(and therefore more trustworthy) n, for free. Running a separate bigger
parity check first would only buy "the harness is probably right" without
producing the Gate-1 number itself -- doing the same work twice.

**Sizing the measurement itself needs real timing, not a guess**, because
failed episodes are expensive: they run the full `max_steps` cap (520)
where a success finishes in ~75 -- roughly 7x. At a 40-60% failure rate,
most wall-clock time goes to episodes that don't succeed, so `total_run_s`
scales with the failure rate, not just episode count. `run_policy_rollout.py`
now reports `total_run_s` / `avg_success_episode_s` / `avg_fail_episode_s`
in `rollout_summary.json` (added after the smoke run above already
completed -- re-running that same config with `resume: true` costs nothing
new, since every episode is already stored, and recomputes these numbers
from each episode's already-saved `execution_time`). The actual 3-5-task ×
15-20-episode config is sized from those real numbers once available, not
guessed.

**Real timing (from the resumed smoke run): `avg_success_episode_s=89.1`,
`avg_fail_episode_s=493.6`** (~5.5x, confirming the failed-episodes-
dominate-cost prediction). At this checkpoint's 40-60% failure rate on
`libero_spatial`, expected cost is ~4.2-5.5 min/episode.

**Decision: 3 tasks × 20 episodes = 60 episodes (~4.2-5.5 hours).**
`configs/policy_rollout_libero_spatial_gate1.yaml` -- `task_ids: [0, 1, 2]`.
Task 0's own 20-episode subset also serves as the harness-parity check
against `lerobot-eval`'s 3/5 reference, at 4x the smoke run's n. Episode-
level incremental save + resume (`EpisodeStore`, already built for exactly
this Colab-survival case) means a mid-run disconnect loses at most the one
in-flight episode -- re-running the same command continues, not restarts.

```
!python {XGAP_DRIVE_ROOT}/scripts/run_policy_rollout.py \
    --config {XGAP_DRIVE_ROOT}/configs/policy_rollout_libero_spatial_gate1.yaml
```

## Test 2: state save/restore determinism

Flagged in this project's own planning as the hardest implementation
element, untouched until now: does "same state + same action = same
result" actually hold? This underlies every future Oracle/World-model/
Random candidate comparison (`n_decision_points`/`exec_horizon`/
`selection_unit`, see "Design constraints" below) -- if restoring a saved
simulator state and replaying the same action doesn't reproduce the same
outcome, comparing candidates branched from a shared decision point is
meaningless. **Run this BEFORE the Gate-1 measurement, not in parallel
with it** -- both need the GPU (Colab gives one), and this is the
cheaper, more urgent question: a bad result here could mean the Gate-1
number itself isn't as precise as assumed (episodes near a decision
boundary could flip on restore noise), so it's worth knowing first. If
the Gate-1 cell is already running, stop it -- `resume: true` means
nothing already completed is lost, re-running the same cell later
continues rather than restarts.

Added `harness.get_sim_state`/`harness.restore_sim_state`
(`env.sim.get_state().flatten()` / `sim.set_state_from_flattened()` +
`sim.forward()` -- the standard robosuite/MuJoCo idiom, and the exact
method name this project already documented, but never implemented, as
the fallback path for precise init-state control -- see `dataset_io.py`'s
module docstring). **Unverified against source for this specific
installed robosuite/MuJoCo version** -- flagged in both functions'
docstrings as the first real test of this API in this project, same
discipline as every other real-Colab-only piece of code here. Also flags
a distinction worth checking empirically rather than assuming: this only
restores MuJoCo's own physics state (qpos/qvel/time/act), not necessarily
any of robosuite/LIBERO's Python-level episode bookkeeping on top of it
(step counters, cumulative reward, success-flag state) -- whether that
gap actually matters is exactly what this test is for.

`scripts/test_state_restore_determinism.py` (+ pure comparison logic in
`xgap_code/state_determinism.py`, tested locally without a simulator --
`tests/test_state_determinism.py`): per trial, reset + `_N_WARMUP_STEPS`
arbitrary actions (to get away from the trivially-matching exact reset
state), save state, branch A steps one fixed non-trivial test action,
restore the saved state, branch B steps the SAME action again, compare A
vs B (5-dim eef+gripper state, and the rendered image -- reported as a
diagnostic max-diff/percent-differing-pixels rather than a strict
pass/fail, since rendering can have tiny nondeterminism even given
identical physics).

```
!python {XGAP_DRIVE_ROOT}/scripts/test_state_restore_determinism.py \
    --task-suite libero_spatial --task-id 0 --n-trials 5
```

- **State identical across all trials** -> the save/restore mechanism is
  trustworthy for candidate branching; proceed to building Oracle/World-
  model/Random condition logic on top of it.
- **State differs** -> `n_action_steps`-shaped candidate comparisons would
  be comparing noise, not real differences -- needs root-causing (missing
  bookkeeping restore, a stochastic op somewhere in the physics/policy
  path, or the `get_sim_state`/`restore_sim_state` API assumption itself
  being wrong for this installed version) before any Oracle/World-model/
  Random work starts.

**Result: not bit-identical, but small and consistent.** All 5 trials:
state `max_abs_diff` ~0.0014-0.0016 (tight clustering across trials, not
random-magnitude noise), image ~0.18-0.22% of pixels differing, **reward
identical in all 5**. Most likely explanation: mujoco_py's `MjSimState`
(`time`, `qpos`, `qvel`, `act`) does not include the contact solver's
warm-start acceleration (`qacc_warmstart`) -- physics is deterministic
given the TRUE internal solver state, but this flattened snapshot may not
capture all of it, so the very next physics step after a restore can
converge slightly differently even from "the same" saved state.

**The single-step result's reward-matched-5/5 is not reassuring on its
own.** LIBERO's success signal is binary/coarse -- a few mm of object
displacement doesn't register at all -- so it cannot detect whether the
observed residual matters. **The only metric that actually matters: is
restore-noise small RELATIVE TO the state divergence a genuinely
different candidate action would produce**, not small in some absolute
sense. Rewrote `test_state_restore_determinism.py` to measure both
directly from the same saved state:

- **signal**: two DIFFERENT fixed action chunks (`chunk_1`, `chunk_2`,
  distinct RNG streams -- as if two different policy/world-model
  proposals from the same decision point) run for `--n-compare-steps`
  each; the resulting state divergence between them is what a real
  candidate comparison would be measuring.
- **noise**: `chunk_1` run twice from the same saved state (once as
  branch A, restore, then again as branch C) -- the resulting divergence
  is restore-imprecision alone, same action chosen both times.

`ratio = signal / noise` at every step. Threshold this project is using
(from the same planning that flagged this as the hardest implementation
element): **~100x is practically safe, ~10x is risky, ~1x means candidate
comparison isn't meaningful at all** -- this ratio, not either raw
number, is Test 2's actual pass/fail criterion.

```
!python {XGAP_DRIVE_ROOT}/scripts/test_state_restore_determinism.py \
    --task-suite libero_spatial --task-id 0 --n-trials 3 --n-compare-steps 10
```

- **min ratio ≥ 100x** -> practically safe; proceed to building
  Oracle/World-model/Random condition logic on the save/restore mechanism
  as-is.
- **min ratio 10-100x** -> risky -- candidate comparisons whose real
  signal happens to be small could be unreliable; worth root-causing
  before depending on it (most likely candidate: finding a way to also
  save/restore `qacc_warmstart`, or whatever else `MjSimState` omits).
- **min ratio < 10x** -> candidate comparison does not work at this noise
  level -- must be fixed before any Oracle/World-model/Random work
  starts.

## Design constraints encoded in this codebase (do not violate)

- `n_decision_points` (config field `n_decision_points`, default `1`) must be applied identically across Oracle / World-model / Random conditions — enforced in code, not by convention.
- `exec_horizon` (how many env steps run per policy call) and `selection_unit` (granularity at which candidates are compared) are separate config fields even before `selection_unit` has more than one implemented value. Step-level selection would break chunk-internal structure and produce a fake performance drop — this is a documented invariant, not a hypothetical.
- No monkey-patching `lerobot` or `libero`. If ever unavoidable, isolate in its own module with a comment explaining why.
- All experiment parameters live in `configs/*.yaml`. No hardcoded constants in `xgap_code/` or `scripts/`.
