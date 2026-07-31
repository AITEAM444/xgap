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

**Dead end, documented so it isn't re-attempted:** `meta/episodes/*.parquet`'s
`data/chunk_index` / `data/file_index` columns (meant to map an episode to its
shard file) are **stale** for this dataset as currently hosted — they claim
episodes 0-14 all live in `chunk-000/file-000.parquet`; that file on disk only
actually contains episodes 0-2. `xgap_code/dataset_io.py` therefore goes
through lerobot's own `LeRobotDataset` (not yet runnable this session — see
below) rather than re-deriving shard locations from that metadata, and
`scripts/analyze_demo_actions.py` takes shard paths as explicit arguments
instead of resolving them automatically.

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

### Next action (needs Colab)

Run `python scripts/run_demo_replay.py --config configs/demo_replay.yaml`
(without `--mock`) in Colab after `setup_colab.sh`. Per the working decision
this session: dual-run `relative`/`absolute`, adopt whichever has higher
success as the confirmed env/demo convention (not a sweep axis — H4 stays in
step 4), and if both are low the suspect is init-state/episode-index wiring,
not `control_mode` — `decide_env_convention()` already encodes this branch and
writes it to `control_mode_decision.json`. Stop and report per the original
instruction once this has run for real.

## Design constraints encoded in this codebase (do not violate)

- `n_decision_points` (config field `n_decision_points`, default `1`) must be applied identically across Oracle / World-model / Random conditions — enforced in code, not by convention.
- `exec_horizon` (how many env steps run per policy call) and `selection_unit` (granularity at which candidates are compared) are separate config fields even before `selection_unit` has more than one implemented value. Step-level selection would break chunk-internal structure and produce a fake performance drop — this is a documented invariant, not a hypothetical.
- No monkey-patching `lerobot` or `libero`. If ever unavoidable, isolate in its own module with a comment explaining why.
- All experiment parameters live in `configs/*.yaml`. No hardcoded constants in `xgap_code/` or `scripts/`.
