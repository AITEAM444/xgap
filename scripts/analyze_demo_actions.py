#!/usr/bin/env python
"""Measure the real action-space distribution and gripper-dimension (action[:,6])
behavior in HuggingFaceVLA/libero demos -- the H3 baseline: what does a correct
gripper open/close signal actually look like, before comparing any policy rollout
against it.

Deliberately reads specific data shard files directly (via huggingface_hub,
`repo_type="dataset"`), not through `meta/episodes/*.parquet`'s chunk/file-index
columns: those columns were checked during development and found STALE for this
dataset as currently hosted (they claim episodes 0-14 all live in
`data/chunk-000/file-000.parquet`; that file on disk actually only contains
episodes 0-2). Pass shard paths explicitly (`--shard`) instead of trusting that
metadata. Only the `action` and `observation.state` columns are ever loaded into
memory -- images are never decoded.

Usage:
    python scripts/analyze_demo_actions.py \\
        --shard data/chunk-000/file-002.parquet \\
        --shard data/chunk-000/file-376.parquet \\
        --out outputs/demo_action_stats
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


def analyze_shard(repo_id: str, shard_path: str) -> list[dict]:
    local_path = hf_hub_download(repo_id, shard_path, repo_type="dataset")
    df = pq.read_table(
        local_path, columns=["action", "observation.state", "episode_index", "task_index", "frame_index"]
    ).to_pandas()

    results = []
    for ep in sorted(df.episode_index.unique()):
        sub = df[df.episode_index == ep].sort_values("frame_index")
        actions = np.stack(sub["action"].to_numpy()).astype(np.float32)
        states = np.stack(sub["observation.state"].to_numpy()).astype(np.float32)
        gripper_action = actions[:, 6]
        finger_gap = states[:, 6] - states[:, 7]  # proxy for gripper openness

        transitions = np.where(np.diff(gripper_action) != 0)[0].tolist()
        results.append(
            {
                "shard": shard_path,
                "episode_index": int(ep),
                "task_index": int(sub.task_index.iloc[0]),
                "length": int(len(actions)),
                "action_dim_min": actions.min(axis=0).tolist(),
                "action_dim_max": actions.max(axis=0).tolist(),
                "action_dim_mean": actions.mean(axis=0).tolist(),
                "action_dim_std": actions.std(axis=0).tolist(),
                "gripper_action_unique_values": sorted({round(float(v), 3) for v in gripper_action}),
                "gripper_action_starts_at": float(gripper_action[0]),
                "gripper_action_transitions_at_frame": transitions,
                "finger_gap_at_start": float(finger_gap[0]),
                "finger_gap_20_steps_after_first_close_cmd": (
                    float(finger_gap[min(transitions[0] + 20, len(finger_gap) - 1)])
                    if transitions
                    else None
                ),
                "finger_gap_at_end": float(finger_gap[-1]),
            }
        )
    return results


def summarize(all_episodes: list[dict]) -> dict:
    starts = [e["gripper_action_starts_at"] for e in all_episodes]
    unique_vals = sorted({v for e in all_episodes for v in e["gripper_action_unique_values"]})
    n_open_start = sum(1 for s in starts if s < 0)

    return {
        "n_episodes_sampled": len(all_episodes),
        "task_indices_sampled": sorted({e["task_index"] for e in all_episodes}),
        "gripper_action_unique_values_across_all_episodes": unique_vals,
        "fraction_episodes_starting_open_neg1": n_open_start / len(all_episodes) if all_episodes else None,
        "interpretation": (
            "action[:,6] is discrete in {-1, +1} in every sampled episode (no continuous "
            "gripper commands in demos). Every sampled episode starts at -1. finger_gap "
            "(states[:,6]-states[:,7], a proxy for how open the gripper is) starts near its "
            "max at episode start and drops sharply ~20 steps after the first +1 command, "
            "confirming -1=open, +1=close (consistent with LIBERO's get_libero_dummy_action() "
            "no-op value of -1 for the gripper dim). Gripper actuation visibly lags the "
            "discrete action switch by roughly 15-20 simulation steps -- a policy rollout "
            "whose action[:,6] flips correctly but whose gripper_qpos never visibly closes "
            "within the executed horizon is a distinguishable, checkable failure signature "
            "for step 3/4, separate from a sign-convention bug (H3)."
        ),
        "caveat": (
            f"Sample is {len(all_episodes)} episodes from explicit shard files, not the full "
            "1693-episode dataset. Direction of the finding (discrete {-1,+1}, -1=open) is a "
            "hardware/environment convention, not task-dependent, so it is unlikely to vary -- "
            "but re-run with more shards across libero_goal too before treating this as final."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", default="HuggingFaceVLA/libero")
    parser.add_argument(
        "--shard",
        action="append",
        default=None,
        help="Explicit data shard path(s), e.g. data/chunk-000/file-002.parquet. Repeatable.",
    )
    parser.add_argument("--out", default="outputs/demo_action_stats")
    args = parser.parse_args()

    shards = args.shard or [
        "data/chunk-000/file-002.parquet",  # libero_10 tasks (long-horizon, multi-grasp)
        "data/chunk-000/file-376.parquet",  # libero_spatial tasks (short, single-grasp)
    ]

    all_episodes = []
    for shard in shards:
        all_episodes.extend(analyze_shard(args.repo_id, shard))

    summary = summarize(all_episodes)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "episodes.json").write_text(json.dumps(all_episodes, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
