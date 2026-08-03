"""Tests for the shard-scanning demo-episode loader. `load_task_demo_episodes`
itself needs lerobot (for LeRobotDatasetMetadata) and isn't testable here --
see README "Local environment check" -- but `scan_shards_for_episodes` only
needs `huggingface_hub` + `pyarrow`, both available locally, so it gets real
test coverage against a synthetic fixture shard instead of another blind
Colab round-trip."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from xgap_code.dataset_io import scan_shards_for_episodes


def _write_fixture_shard(path, episodes: dict[int, int]):
    """episodes: {episode_index: n_frames}. Writes action/observation.state/
    episode_index/frame_index columns only -- no image columns, matching what
    scan_shards_for_episodes actually projects (and confirming it tolerates a
    file that never had image columns at all, as well as one that does)."""
    rows = {"action": [], "observation.state": [], "episode_index": [], "frame_index": []}
    for ep_idx, n_frames in episodes.items():
        for f in range(n_frames):
            rows["action"].append([float(f)] * 7)
            rows["observation.state"].append([float(f)] * 8)
            rows["episode_index"].append(ep_idx)
            rows["frame_index"].append(f)
    table = pa.table(rows)
    pq.write_table(table, path)


def test_scan_finds_episodes_in_a_single_shard(tmp_path, monkeypatch):
    shard = tmp_path / "file-000.parquet"
    _write_fixture_shard(shard, {6: 5, 7: 3})

    def fake_hf_hub_download(repo_id, path, repo_type, cache_dir=None):
        return str(shard)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)

    result = scan_shards_for_episodes("fake/repo", {6, 7}, shard_paths=["data/chunk-000/file-000.parquet"])

    assert set(result.keys()) == {6, 7}
    assert result[6]["actions"].shape == (5, 7)
    assert result[7]["actions"].shape == (3, 7)
    assert result[6]["states"].shape == (5, 8)
    np.testing.assert_allclose(result[6]["actions"][:, 0], [0, 1, 2, 3, 4])


def test_scan_stops_early_once_all_targets_found(tmp_path, monkeypatch):
    shard_a = tmp_path / "file-000.parquet"
    shard_b = tmp_path / "file-001.parquet"
    _write_fixture_shard(shard_a, {0: 2})
    _write_fixture_shard(shard_b, {1: 2})

    calls = []

    def fake_hf_hub_download(repo_id, path, repo_type, cache_dir=None):
        calls.append(path)
        return str(shard_a) if "file-000" in path else str(shard_b)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)

    result = scan_shards_for_episodes(
        "fake/repo", {0}, shard_paths=["data/chunk-000/file-000.parquet", "data/chunk-000/file-001.parquet"]
    )

    assert set(result.keys()) == {0}
    assert calls == ["data/chunk-000/file-000.parquet"]  # never touched file-001


def test_scan_raises_clearly_when_episode_never_found(tmp_path, monkeypatch):
    shard = tmp_path / "file-000.parquet"
    _write_fixture_shard(shard, {0: 2})

    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", lambda repo_id, path, repo_type, cache_dir=None: str(shard)
    )

    with pytest.raises(ValueError, match="could not locate"):
        scan_shards_for_episodes("fake/repo", {0, 999}, shard_paths=["data/chunk-000/file-000.parquet"])


def test_scan_frames_sorted_by_frame_index_not_file_order(tmp_path, monkeypatch):
    """Write frames out of order on disk; scan must still return them sorted."""
    rows = {
        "action": [[3.0] * 7, [1.0] * 7, [2.0] * 7],
        "observation.state": [[0.0] * 8] * 3,
        "episode_index": [5, 5, 5],
        "frame_index": [3, 1, 2],
    }
    shard = tmp_path / "file-000.parquet"
    pq.write_table(pa.table(rows), shard)

    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", lambda repo_id, path, repo_type, cache_dir=None: str(shard)
    )

    result = scan_shards_for_episodes("fake/repo", {5}, shard_paths=["data/chunk-000/file-000.parquet"])
    np.testing.assert_allclose(result[5]["actions"][:, 0], [1.0, 2.0, 3.0])
