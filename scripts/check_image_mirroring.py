#!/usr/bin/env python
"""Check whether the image tensor actually fed to policy.select_action() is
left-right mirrored, for BOTH LIBERO cameras (image=agentview, image2=wrist),
against a reference frame from the training dataset.

Two independently suspect paths that must not be conflated:
  1. The saved eval .mp4 (env.render() -> write_video) -- this is what's been
     watched so far.
  2. The actual tensor handed to policy.select_action() -- built via
     preprocess_observation() -> env_preprocessor() -> preprocessor(), a
     DIFFERENT code path from (1). A flip in one does not imply a flip in
     the other.

This script only touches (2): it builds the observation the exact same way
lerobot's own `rollout()` does (see lerobot/scripts/lerobot_eval.py source --
same public functions, same order: make_env, make_policy,
make_pre_post_processors, make_env_pre_post_processors,
preprocess_observation, env_preprocessor, preprocessor), for one env.reset()
with no stepping, and dumps the resulting per-camera tensors as PNG.

Config construction reuses lerobot's own `@parser.wrap()` decorator (see
`lerobot/configs/parser.py`), not a raw `draccus.parse(...)` call -- a plain
draccus call doesn't know how to resolve `--policy.path=...` into a concrete
policy config (that CLI-only field is stripped and resolved via a separate
`sys.argv`-reading code path inside `wrap()` itself, confirmed by reading
its source; a naive re-implementation missed this and errored with
"unrecognized arguments: --policy.path=..."). Reusing the real decorator on
a stand-in function that just returns `cfg` guarantees identical parsing to
every `!lerobot-eval ...` command run so far, at the cost of temporarily
overwriting `sys.argv` (restored immediately after). Still no lerobot/libero
monkey-patching -- `wrap()` is a public, intended-for-reuse decorator, not
something being patched.

The reference frame comes from xgap_code.dataset_io.fetch_reference_frame_images
(same libero_10 task, same demo episode used throughout this project,
frame_index=0) -- comparison basis is "same orientation as training data",
not "looks upright to a human".

    python scripts/check_image_mirroring.py --out-dir /content/outputs/mirror_check
"""

import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from lerobot.configs import parser as lerobot_parser  # noqa: E402
from lerobot.configs.eval import EvalPipelineConfig  # noqa: E402
from lerobot.envs import make_env, make_env_pre_post_processors, preprocess_observation  # noqa: E402
from lerobot.policies import make_policy, make_pre_post_processors  # noqa: E402

from xgap_code.dataset_io import fetch_reference_frame_images  # noqa: E402
from xgap_code.harness import get_libero_task_language  # noqa: E402

# The same demo used throughout this project's comparisons (libero_10 task 0,
# dataset-global episode_index=8) -- see README.
_REFERENCE_TASK_SUITE = "libero_10"
_REFERENCE_TASK_ID = 0
_REFERENCE_EPISODE_INDEX = 8
_REFERENCE_REPO_ID = "HuggingFaceVLA/libero"


def _build_eval_cfg(out_dir: Path) -> EvalPipelineConfig:
    """Build EvalPipelineConfig via the real `@parser.wrap()` machinery (see
    module docstring for why a raw draccus.parse(...) call is not equivalent),
    by temporarily pointing sys.argv at the same args every real
    `!lerobot-eval ...` command in this project has used, then restoring it."""

    @lerobot_parser.wrap()
    def _capture(cfg: EvalPipelineConfig) -> EvalPipelineConfig:
        return cfg

    saved_argv = sys.argv
    sys.argv = [
        "lerobot-eval",
        "--policy.path=HuggingFaceVLA/smolvla_libero",
        "--policy.n_action_steps=10",
        "--env.type=libero",
        f"--env.task={_REFERENCE_TASK_SUITE}",
        f"--env.task_ids=[{_REFERENCE_TASK_ID}]",
        "--eval.batch_size=1",
        "--eval.n_episodes=1",
        "--env.max_parallel_tasks=1",
        f"--output_dir={out_dir}",
    ]
    try:
        return _capture()
    finally:
        sys.argv = saved_argv


def _save_tensor_as_png(tensor, path: Path) -> None:
    """tensor: (C,H,W), whatever values the policy actually receives (may be
    normalized) -- rescaled per-channel min/max to 0-255 purely for visual
    orientation inspection, not for exact pixel-value comparison."""
    arr = tensor.detach().to("cpu").float().numpy()
    arr = np.transpose(arr, (1, 2, 0))  # C,H,W -> H,W,C
    arr = arr - arr.min()
    maxval = arr.max()
    if maxval > 0:
        arr = arr / maxval
    arr = (arr * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def dump_policy_input_images(out_dir: Path) -> list[str]:
    cfg = _build_eval_cfg(out_dir)

    envs = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs, trust_remote_code=cfg.trust_remote_code)
    env = next(iter(next(iter(envs.values())).values()))

    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env, rename_map=cfg.rename_map)
    policy.eval()

    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }
    preprocessor, _postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )
    env_preprocessor, _env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)

    observation, _info = env.reset(seed=[0])
    observation = preprocess_observation(observation)
    try:
        observation["task"] = list(env.call("task_description"))
    except (AttributeError, NotImplementedError):
        observation["task"] = [""] * env.num_envs
    observation = env_preprocessor(observation)
    observation = preprocessor(observation)

    image_keys = [k for k in observation if "image" in k.lower()]
    saved = []
    for key in image_keys:
        tensor = observation[key][0]  # first (only) env in the batch
        safe_name = key.replace(".", "_")
        path = out_dir / f"policy_input_{safe_name}.png"
        _save_tensor_as_png(tensor, path)
        saved.append(str(path))
        print(f"policy input -- {key}: shape={tuple(tensor.shape)} -> {path}")

    env.close()
    return saved


def dump_reference_images(out_dir: Path) -> list[str]:
    task_language = get_libero_task_language(_REFERENCE_TASK_SUITE, _REFERENCE_TASK_ID)
    print(f"reference task: '{task_language}' ({_REFERENCE_TASK_SUITE}:{_REFERENCE_TASK_ID})")
    images = fetch_reference_frame_images(_REFERENCE_REPO_ID, _REFERENCE_EPISODE_INDEX, frame_index=0)
    saved = []
    for key, raw_bytes in images.items():
        safe_name = key.replace(".", "_")
        path = out_dir / f"dataset_reference_{safe_name}.png"
        Image.open(io.BytesIO(raw_bytes)).save(path)
        saved.append(str(path))
        print(f"dataset reference -- {key} -> {path}")
    return saved


def main() -> None:
    arg_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    arg_parser.add_argument("--out-dir", required=True)
    args = arg_parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dump_reference_images(out_dir)
    dump_policy_input_images(out_dir)


if __name__ == "__main__":
    main()
