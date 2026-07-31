"""Read demo action/state sequences from a LeRobot-format HF dataset
(HuggingFaceVLA/libero by default) for a specific task.

Uses lerobot's own `LeRobotDataset` (lazy import) rather than hand-rolling
parquet-shard discovery. This was a deliberate reversal during development:
an earlier version of this module tried to resolve episode -> shard-file
location directly from `meta/episodes/*.parquet` (`data/chunk_index`,
`data/file_index` columns). That metadata turned out to be STALE for
HuggingFaceVLA/libero as currently hosted -- e.g. it claims episodes 0-14 all
live in `data/chunk-000/file-000.parquet`, but that file on disk actually only
contains episodes 0-2. Trusting it would have silently loaded the wrong
episodes. `LeRobotDataset` resolves this correctly internally, so we go
through it instead of re-deriving shard locations ourselves.

CAVEAT: the exact attribute names below (`dataset.meta.episodes`,
`episode_filter`, `dataset.hf_dataset`) were confirmed against lerobot's
`main` branch source at the time of writing, but were not runnable in this
session (no lerobot installed locally -- see README "Local environment
check"). Re-verify against the actually-installed version before trusting
this path in Colab; if names drifted, the ad hoc raw-parquet analysis in
`scripts/analyze_demo_actions.py`'s fallback mode (`--no-lerobot`) is the
already-verified escape hatch used to produce this session's H3 baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DemoEpisode:
    episode_index: int  # dataset-global index
    task_index: int
    task_name: str
    within_task_index: int  # position among this task's episodes, in dataset (episode_index) order
    actions: np.ndarray  # (T, 7) float32
    states: np.ndarray  # (T, 8) float32


def load_task_demo_episodes(
    repo_id: str, task_name: str, max_episodes: int | None = None
) -> list[DemoEpisode]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # lazy import, see module docstring

    ds = LeRobotDataset(repo_id, episode_filter=lambda ep: ep.get("tasks", [None])[0] == task_name)
    task_indices = sorted(ds.meta.episodes["episode_index"].tolist())
    # NOTE: within_task_index assumes dataset (episode_index) order matches LIBERO's own
    # init-state ordering for this task. Not verified -- see harness.make_real_libero_env docstring.
    episodes: list[DemoEpisode] = []
    for within_idx, ep_idx in enumerate(task_indices):
        if max_episodes is not None and within_idx >= max_episodes:
            break
        ep_frames = ds.hf_dataset.filter(lambda row: row["episode_index"] == ep_idx)
        actions = np.stack([np.asarray(a, dtype=np.float32) for a in ep_frames["action"]])
        states = np.stack([np.asarray(s, dtype=np.float32) for s in ep_frames["observation.state"]])
        task_index = int(ep_frames["task_index"][0])
        episodes.append(
            DemoEpisode(
                episode_index=int(ep_idx),
                task_index=task_index,
                task_name=task_name,
                within_task_index=within_idx,
                actions=actions,
                states=states,
            )
        )
    return episodes
