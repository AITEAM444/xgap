#!/usr/bin/env python
"""Compute longest gripper-close run per episode from a `lerobot-eval --eval.recording=true`
output directory.

`lerobot-eval`'s own `eval_policy()`/`rollout()` (see `lerobot/scripts/lerobot_eval.py`) write a
real LeRobotDataset -- with raw per-step `action` -- to `<output_dir>/recordings/<task_group>_<task_id>/`
whenever `--eval.recording=true` is passed, instead of only saving videos. That dataset uses the
same on-disk shard layout (`data/chunk-*/file-*.parquet`) as HuggingFaceVLA/libero, so this reads
it exactly the way `xgap_code/dataset_io.py` reads real demo shards -- column-projected, no
LeRobotDataset/image materialization -- just pointed at local files instead of hf_hub_download.

    python scripts/gripper_close_from_recording.py --recording-dir <output_dir>/recordings/libero_10_0
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow.parquet as pq  # noqa: E402

from xgap_code.gripper_metrics import longest_close_run  # noqa: E402


def run(recording_dir: Path) -> list[dict]:
    shard_paths = sorted(glob.glob(str(recording_dir / "data" / "chunk-*" / "file-*.parquet")))
    if not shard_paths:
        raise ValueError(f"No parquet shards found under {recording_dir}/data/")

    by_episode: dict[int, list] = {}
    for shard_path in shard_paths:
        table = pq.read_table(shard_path, columns=["action", "episode_index"])
        for ep, action in zip(table.column("episode_index").to_pylist(), table.column("action").to_pylist()):
            by_episode.setdefault(ep, []).append(action)

    return [
        {
            "episode_index": ep,
            "n_steps": len(actions),
            "longest_close_run_steps": longest_close_run(actions),
        }
        for ep, actions in sorted(by_episode.items())
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recording-dir", required=True)
    args = parser.parse_args()

    for row in run(Path(args.recording_dir)):
        print(row)


if __name__ == "__main__":
    main()
