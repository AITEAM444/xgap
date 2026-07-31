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

CORRECTION 3 (a third Colab run): `ValueError: need at least one array to
stack` -- the fix above pre-seeded `frames_by_episode` keys from
`ds.episodes` (`filter_episodes()`'s sorted, dataset-global episode indices),
then only kept frames whose `episode_index` matched one of those keys. If the
filtered `LeRobotDataset`'s own `__getitem__` reports `episode_index` in a
different numbering than `filter_episodes()` returns (never actually
confirmed either way from source), every frame gets silently dropped and an
episode ends up with zero frames. Fixed to build the grouping purely from
`episode_index` values actually observed on `ds[i]`, with no cross-referenced
assumption about a second accessor using the same numbering.

CORRECTION 4 (diagnosed directly, not from another blind Colab retry): after
correction 3, `filter_episodes()` correctly identified all 33 episodes for a
test task, but only 2 of them (out of 33) ended up with any retrievable
frames. Root cause, confirmed from lerobot's actual source, is a REAL bug/data
problem upstream of our code, not fixable by changing our own filtering
logic: `LeRobotDatasetMetadata.get_data_file_path()`
(`src/lerobot/datasets/dataset_metadata.py`) resolves which shard file an
episode's data lives in via `meta/episodes/*.parquet`'s `data/chunk_index` /
`data/file_index` columns -- exactly the columns already found to be STALE
for HuggingFaceVLA/libero earlier this session (episode_index 8's row claims
`data/chunk-000/file-000.parquet`; that file's real contents are only
episodes 0-2). `DatasetReader.get_episodes_file_paths()`
(`src/lerobot/datasets/dataset_reader.py`) uses this to build the
`allow_patterns` list for a SELECTIVE `snapshot_download` when constructing
`LeRobotDataset` with `episode_filter`/`episodes` set -- so for most requested
episodes, the wrong (or an incomplete) shard file gets fetched, and that
episode's frames silently never materialize. This is not something to work
around by monkey-patching lerobot (against project principle) -- worked
around instead by forcing a full, non-selective download of the dataset's
`data/` directory into the SAME cache directory lerobot's own downloader
reads from, before ever constructing `LeRobotDataset` with a filter. See
`ensure_full_dataset_cached()` below.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=None)
def ensure_full_dataset_cached(repo_id: str) -> None:
    """Force a full download of `repo_id`'s `data/` (and `meta/`) directory into
    lerobot's own Hub cache, bypassing `LeRobotDataset`'s normal selective
    per-episode download.

    See CORRECTION 4 in this module's docstring for why this is necessary: the
    selective downloader trusts stale shard-location metadata and silently
    fetches the wrong files for most episodes. Downloading everything up front
    means whichever (possibly wrong) files lerobot's own logic later asks for,
    they are already present locally -- nothing is silently missing.

    Uses `HF_LEROBOT_HUB_CACHE` (lerobot's actual cache dir, from
    `lerobot.utils.constants` -- NOT a path we invent) as `cache_dir`, so this
    pre-download lands exactly where `LeRobotDataset` will look for it.
    `@lru_cache` avoids re-verifying the full manifest on every call within one
    process (`snapshot_download` itself is already incremental/idempotent
    across process runs -- already-downloaded files aren't re-fetched).

    Cost: this is a real, large download (tens of GB for HuggingFaceVLA/libero
    -- both `data/chunk-000` and `data/chunk-001`, since episodes for a single
    task/suite are NOT contiguous within a chunk -- see the "suite blocking"
    note in the analysis this session). Point `HF_LEROBOT_HOME` at Drive (see
    setup_colab.sh) so this is a one-time cost, not repeated every session.
    """
    from huggingface_hub import snapshot_download
    from lerobot.utils.constants import HF_LEROBOT_HUB_CACHE

    snapshot_download(
        repo_id,
        repo_type="dataset",
        cache_dir=str(HF_LEROBOT_HUB_CACHE),
        allow_patterns=["data/**", "meta/**"],
    )


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

    # Work around stale shard-location metadata -- see CORRECTION 4 / ensure_full_dataset_cached's
    # docstring. Must happen before LeRobotDataset(episode_filter=...) below.
    ensure_full_dataset_cached(repo_id)

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

    ds = LeRobotDataset(repo_id, episode_filter=lambda ep: ep["tasks"][0] == task_name)

    # Group by episode_index as it ACTUALLY appears on the frames returned by ds[i], rather
    # than pre-seeding keys from ds.episodes (filter_episodes()'s return value) and hoping the
    # two numberings agree. An earlier version did the latter and crashed with `ValueError:
    # need at least one array to stack` on the very next real run -- ds.episodes lists
    # dataset-global indices, but there's no source-confirmed guarantee the (possibly
    # episode_filter-scoped) dataset's own __getitem__ reports episode_index in that same
    # numbering. Building the grouping purely from observed frame data sidesteps that
    # question entirely; within_task_index ordering below assumes dataset-global
    # (episode_index) order matches LIBERO's own init-state collection order for this task --
    # still a separate, unverified assumption, see harness.make_real_libero_env's docstring.
    frames_by_episode: dict[int, list[dict]] = {}
    for i in range(len(ds)):
        frame = ds[i]
        ep_idx = _to_scalar(frame["episode_index"])
        frames_by_episode.setdefault(ep_idx, []).append(frame)

    if not frames_by_episode:
        raise ValueError(
            f"episode_filter matched task '{task_name}' but the resulting dataset has 0 "
            f"frames (repo_id='{repo_id}'). This means the filter and the loaded data "
            f"disagree -- report this, do not silently retry with a different task."
        )

    episode_indices = sorted(frames_by_episode.keys())
    if max_episodes is not None:
        episode_indices = episode_indices[:max_episodes]

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
