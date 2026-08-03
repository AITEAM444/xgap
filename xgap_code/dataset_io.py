"""Read demo action/state sequences from a LeRobot-format HF dataset
(HuggingFaceVLA/libero by default) for a specific task.

CORRECTION 1 (caught by an actual Colab run): the first version of
`load_task_demo_episodes` filtered with `ep.get("tasks", [None])[0] ==
task_name`, where `task_name` was accidentally our own "suite:task_id"
bookkeeping label instead of LIBERO's natural-language instruction string --
matched nothing, `ValueError: The episode filter did not match any episode`.

CORRECTION 2 (a second Colab run, right after "fixing" correction 1): assumed
-- from `lerobot_dataset.py`'s `episode_filter` docstring, which explicitly
lists `task_index` as an available field -- that the right key was
`task_index`. That crashed with `KeyError: 'task_index'`: the docstring is
wrong (or describes a different code path). `load_episodes()` in
`src/lerobot/datasets/io_utils.py` loads the raw episodes parquet and drops
only `stats/*` columns -- `"tasks"` (a `list[str]`, one entry per episode)
survives untouched. This matches what was already independently confirmed by
downloading real parquet shards directly this session (see
outputs/demo_action_stats/) -- that source should have been trusted over the
docstring from the start.

CORRECTION 3 (a third Colab run): `ValueError: need at least one array to
stack` -- an intermediate version pre-seeded frame groups from
`LeRobotDatasetMetadata.filter_episodes()`'s dataset-global episode indices
and hoped a separately-constructed `LeRobotDataset`'s own `__getitem__`
reported `episode_index` in that same numbering. It didn't reliably.

CORRECTION 4 (diagnosed directly): after correction 3, `filter_episodes()`
correctly identified all 33 episodes for a test task, but only 2 ended up
with retrievable frames. Root cause, confirmed from lerobot's actual source:
`LeRobotDatasetMetadata.get_data_file_path()` resolves an episode's shard
file via `meta/episodes/*.parquet`'s `data/chunk_index` / `data/file_index`
columns -- confirmed STALE for HuggingFaceVLA/libero earlier this session
(episode_index 8's row claims `data/chunk-000/file-000.parquet`; that file's
real contents are only episodes 0-2). `LeRobotDataset`'s selective
`snapshot_download` trusts this and fetches the wrong/incomplete shard set
for most episodes.

CORRECTION 5 (two more real Colab failures, right after "fixing" correction
4): force-downloading the ENTIRE dataset via `snapshot_download` to work
around correction 4 hit two separate infrastructure failures in a row --
first a silent death mid-transfer while targeting Drive (Colab's Drive FUSE
mount is unreliable for sustained large writes; no traceback, no
user-initiated interrupt), then an out-of-memory kill after switching to
local disk. The second failure implicates `LeRobotDataset` itself, not just
the download: constructing it (or accessing `ds[i]` across a fully-cached,
unfiltered-at-the-file-level dataset) appears to materialize far more than
the small numeric columns we actually need -- this dataset embeds images
directly inside the data parquet files (confirmed earlier this session, no
separate video files), so anything that loads full rows pulls image bytes
along with them.

CONCLUSION: `LeRobotDataset`/`LeRobotDatasetMetadata.get_data_file_path()`'s
per-episode shard resolution is not usable for this dataset as currently
hosted, for two independent reasons (stale location metadata, and apparent
full-row materialization cost). This is not fixable by changing our own
filter predicate -- it needed a different approach, not another patch.
`LeRobotDatasetMetadata` (metadata-only: info.json/tasks.parquet/episodes
parquet, no frame data) is still used for task_index resolution and episode
membership, since that part works correctly (see CORRECTION 4 -- the
metadata-level `filter_episodes()` call has never been wrong, only the
per-episode file lookup built on top of it). Everything past that point --
locating and reading actual frame data -- is now done directly against the
Hub's real file listing (not the possibly-stale chunk/file-index columns),
reading ONLY the `action`/`observation.state`/`frame_index`/`episode_index`
columns via `pyarrow` column projection (never touching the embedded image
columns), one shard at a time, discarding each shard's table before moving to
the next -- bounding peak memory to roughly one shard's size regardless of
how many episodes or shards exist. This is the same verified pattern
`scripts/analyze_demo_actions.py` used successfully this session (see
outputs/demo_action_stats/), generalized to scan systematically instead of
against hand-picked example shard paths.
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# Shard downloads MUST NOT default to HF_HOME's cache dir: HF_HOME is deliberately set to
# a Drive path (setup_colab.sh, for small checkpoint/config caching) and streaming
# many-GB shard downloads onto Colab's Drive FUSE mount is exactly what killed an earlier
# approach silently mid-transfer (see module docstring, CORRECTION 5). tempfile.gettempdir()
# resolves to local disk (/tmp or /content-backed) regardless of any env var propagation
# issues between notebook cells -- no new env var to keep in sync required.
_LOCAL_SHARD_CACHE_DIR = str(Path(tempfile.gettempdir()) / "xgap_hf_shard_cache")


@dataclass
class DemoEpisode:
    episode_index: int  # dataset-global index
    task_index: int
    task_name: str
    within_task_index: int  # position among this task's episodes, in dataset (episode_index) order
    actions: np.ndarray  # (T, 7) float32
    states: np.ndarray  # (T, 8) float32


_SHARD_PATH_RE = re.compile(r"^data/chunk-\d+/file-\d+\.parquet$")


def list_shard_paths(repo_id: str) -> list[str]:
    """List the dataset's ACTUAL shard files from the Hub -- not
    meta/episodes/*.parquet's data/chunk_index / data/file_index columns,
    confirmed stale for HuggingFaceVLA/libero (see module docstring)."""
    from huggingface_hub import HfApi

    all_files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    shard_paths = sorted(f for f in all_files if _SHARD_PATH_RE.match(f))
    if not shard_paths:
        raise ValueError(f"No data/chunk-*/file-*.parquet shards found in '{repo_id}'.")
    return shard_paths


def scan_shards_for_episodes(
    repo_id: str, target_episodes: set[int], shard_paths: list[str] | None = None
) -> dict[int, dict[str, np.ndarray]]:
    """Scan shard files one at a time (stopping early once every target episode is
    found), reading ONLY the action/state/index columns via pyarrow column
    projection -- never touching the embedded image columns, which is what keeps
    peak memory bounded to roughly one shard's numeric data regardless of how many
    episodes or shards exist. See module docstring (CORRECTION 5) for why this
    replaced a `LeRobotDataset`-based approach.

    Returns {episode_index: {"actions": (T,7) float32, "states": (T,8) float32}}.
    Raises if any target episode is never found after scanning every shard.
    """
    from huggingface_hub import hf_hub_download

    if shard_paths is None:
        shard_paths = list_shard_paths(repo_id)

    frames_by_episode: dict[int, dict[str, np.ndarray]] = {}
    remaining = set(target_episodes)
    columns = ["action", "observation.state", "episode_index", "frame_index"]

    for shard_path in shard_paths:
        if not remaining:
            break  # every target episode already found -- no need to scan further shards
        local_path = hf_hub_download(
            repo_id, shard_path, repo_type="dataset", cache_dir=_LOCAL_SHARD_CACHE_DIR
        )
        # Column projection: pyarrow never decodes the embedded image columns for rows we
        # don't select, keeping peak memory bounded to this one shard's numeric data.
        table = pq.read_table(local_path, columns=columns)
        ep_col = table.column("episode_index").to_numpy()
        hits = np.isin(ep_col, list(remaining))
        if not hits.any():
            continue
        sub = table.filter(hits).to_pydict()
        sub_eps = np.asarray(sub["episode_index"])
        for ep_idx in {int(e) for e in sub_eps}:
            if ep_idx not in remaining:
                continue
            rows = [i for i, e in enumerate(sub_eps) if int(e) == ep_idx]
            rows.sort(key=lambda i: sub["frame_index"][i])
            frames_by_episode[ep_idx] = {
                "actions": np.asarray([sub["action"][i] for i in rows], dtype=np.float32),
                "states": np.asarray([sub["observation.state"][i] for i in rows], dtype=np.float32),
            }
            remaining.discard(ep_idx)

    if remaining:
        raise ValueError(
            f"Scanned all {len(shard_paths)} shards but could not locate "
            f"{len(remaining)} of {len(target_episodes)} episodes in '{repo_id}': "
            f"missing episode_index {sorted(remaining)[:10]}"
            f"{'...' if len(remaining) > 10 else ''}."
        )
    return frames_by_episode


def load_task_demo_episodes(
    repo_id: str, task_name: str, max_episodes: int | None = None
) -> list[DemoEpisode]:
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    # Metadata-only load (info.json/tasks.parquet/episodes parquet -- no frame or video
    # data) to resolve the task string and which dataset-global episode_index values
    # belong to it. This part of lerobot's metadata handling has been correct in every
    # Colab run so far -- only the per-episode FILE lookup built on top of it (which we
    # no longer use, see module docstring) was the problem.
    meta = LeRobotDatasetMetadata(repo_id)
    task_index = meta.get_task_index(task_name)
    if task_index is None:
        available = list(meta.tasks.index) if meta.tasks is not None else []
        raise ValueError(
            f"Task '{task_name}' not found in dataset '{repo_id}'. "
            f"First few available tasks: {available[:5]}"
        )

    all_target_episodes = sorted(meta.filter_episodes(lambda ep: ep["tasks"][0] == task_name))
    if not all_target_episodes:
        raise ValueError(
            f"No episodes found for task '{task_name}' in dataset '{repo_id}' "
            f"(task_index={task_index}, but filter_episodes matched nothing)."
        )

    # Cap BEFORE scanning, not after: scan_shards_for_episodes stops as soon as every
    # episode it was ASKED for is found, so asking for only the episodes we actually need
    # (e.g. 5 for a smoke config) instead of every episode the task has (e.g. 33+) is what
    # makes it stop early -- capping after the scan means it hunts down every episode for
    # the task regardless of max_episodes, which is what made an earlier run look "stuck"
    # (it wasn't -- it was still correctly finding the other ~28 episodes nobody asked for).
    target_episodes = set(all_target_episodes if max_episodes is None else all_target_episodes[:max_episodes])

    frames_by_episode = scan_shards_for_episodes(repo_id, target_episodes)
    episode_indices = sorted(frames_by_episode.keys())

    return [
        DemoEpisode(
            episode_index=ep_idx,
            task_index=task_index,
            task_name=task_name,
            within_task_index=within_idx,
            actions=frames_by_episode[ep_idx]["actions"],
            states=frames_by_episode[ep_idx]["states"],
        )
        for within_idx, ep_idx in enumerate(episode_indices)
    ]
