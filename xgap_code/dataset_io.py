"""Read demo action/state sequences from a LeRobot-format HF dataset
(HuggingFaceVLA/libero by default) for a specific task.

Uses lerobot's own `LeRobotDataset` / `LeRobotDatasetMetadata` (lazy import)
rather than hand-rolling parquet-shard discovery. This was a deliberate
reversal during development: an earlier version of this module tried to
resolve episode -> shard-file location directly from `meta/episodes/*.parquet`
(`data/chunk_index`, `data/file_index` columns). That metadata turned out to
be STALE for HuggingFaceVLA/libero as currently hosted -- e.g. it claims
episodes 0-14 all live in `data/chunk-000/file-000.parquet`, but that file on
disk actually only contains episodes 0-2. `LeRobotDataset` resolves this
correctly internally, so we go through it instead of re-deriving shard
locations ourselves.

CORRECTION 1 (caught by an actual Colab run): the first version of
`load_task_demo_episodes` filtered with `ep.get("tasks", [None])[0] ==
task_name`, where `task_name` was accidentally our own "suite:task_id"
bookkeeping label instead of LIBERO's natural-language instruction string --
matched nothing, `ValueError: The episode filter did not match any episode`.

CORRECTION 2 (a second Colab run, right after "fixing" correction 1): assumed
-- from `lerobot_dataset.py`'s `episode_filter` docstring, which explicitly
lists `task_index` as an available field -- that the right key was
`task_index`, and changed the filter to `ep["task_index"]`. That crashed with
`KeyError: 'task_index'`. The docstring is simply wrong here (or describes a
different code path): `load_episodes()` in
`src/lerobot/datasets/io_utils.py` loads the raw episodes parquet and drops
only `stats/*` columns -- `"tasks"` (a `list[str]`, one entry per episode)
survives untouched into the dict `episode_filter` sees. This matches what was
already independently confirmed by downloading real parquet shards directly
this session (see outputs/demo_action_stats/) -- that source should have been
trusted over the docstring from the start. Filtering below is back to
`ep["tasks"][0]`, this time compared against the correct value (the actual
LIBERO instruction string, via `harness.get_libero_task_language`) rather than
our internal label.
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


def _to_numpy(x) -> np.ndarray:
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def _to_scalar(x) -> int:
    return int(x.item() if hasattr(x, "item") else x)


def load_task_demo_episodes(
    repo_id: str, task_name: str, max_episodes: int | None = None
) -> list[DemoEpisode]:
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # Metadata-only load (just the meta/ dir -- info.json, tasks.parquet, episodes/*.parquet;
    # no frame or video data) purely to resolve the natural-language task string to this
    # dataset's task_index.
    meta = LeRobotDatasetMetadata(repo_id)
    task_index = meta.get_task_index(task_name)
    if task_index is None:
        available = list(meta.tasks.index) if meta.tasks is not None else []
        raise ValueError(
            f"Task '{task_name}' not found in dataset '{repo_id}'. "
            f"First few available tasks: {available[:5]}"
        )

    # filter_episodes() (dataset_metadata.py) returns episode indices in SORTED order.
    # within_task_index below assumes that sorted dataset-global order matches LIBERO's own
    # init-state collection order for this task -- an assumption, not a fact; see
    # harness.make_real_libero_env's docstring for how to treat it (high replay success is
    # supporting evidence; low success with control_mode/init_states both otherwise correct
    # points back at this mapping).
    ds = LeRobotDataset(repo_id, episode_filter=lambda ep: ep["tasks"][0] == task_name)
    episode_indices = sorted(ds.episodes) if ds.episodes else []
    if max_episodes is not None:
        episode_indices = episode_indices[:max_episodes]

    frames_by_episode: dict[int, list[dict]] = {ep: [] for ep in episode_indices}
    for i in range(len(ds)):
        frame = ds[i]
        ep_idx = _to_scalar(frame["episode_index"])
        if ep_idx in frames_by_episode:
            frames_by_episode[ep_idx].append(frame)

    episodes: list[DemoEpisode] = []
    for within_idx, ep_idx in enumerate(episode_indices):
        frames = sorted(frames_by_episode[ep_idx], key=lambda f: _to_scalar(f["frame_index"]))
        actions = np.stack([_to_numpy(f["action"]) for f in frames]).astype(np.float32)
        states = np.stack([_to_numpy(f["observation.state"]) for f in frames]).astype(np.float32)
        episodes.append(
            DemoEpisode(
                episode_index=ep_idx,
                task_index=task_index,
                task_name=task_name,
                within_task_index=within_idx,
                actions=actions,
                states=states,
            )
        )
    return episodes
