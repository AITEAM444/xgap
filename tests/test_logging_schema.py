"""Smoke tests for the episode store: resume logic, incremental local writes,
batched remote sync, and partial-result tolerance. All run against plain
dataclasses/parquet -- no simulator involved."""

from __future__ import annotations

from xgap_code.logging_schema import EpisodeRecord, EpisodeStore, episode_key


def _dummy_record(control_mode="relative", success=True) -> EpisodeRecord:
    return EpisodeRecord(
        task_id="libero_10:0",
        task_suite="libero_10",
        episode_seed=0,
        environment_seed=0,
        condition=f"demo_replay_{control_mode}",
        action_chunk=[[0.0] * 7, [0.1] * 7],
        episode_success=success,
        rollout_length=2,
        checkpoint_name="N/A_demo_replay",
        checkpoint_hash="N/A_demo_replay",
        git_commit="deadbeef",
        config_file="configs/demo_replay.yaml",
        execution_time=0.5,
        time_inference_s=0.0,
        time_render_s=0.0,
        time_physics_s=0.4,
        time_reset_s=0.1,
        control_mode=control_mode,
        control_freq=10,
    )


def test_write_and_read_roundtrip(tmp_path):
    store = EpisodeStore(str(tmp_path / "local"), remote_root=None, sync_every_n_episodes=100)
    key = episode_key("demo_replay_relative", "libero_10", "libero_10:0", 0)
    store.write_episode(key, _dummy_record())

    table = store.read_all()
    assert table.num_rows == 1
    row = table.to_pylist()[0]
    assert row["task_id"] == "libero_10:0"
    assert row["episode_success"] is True
    # action_chunk is stored as float32 (schema), so compare with float32 round-tripped
    # expectations rather than exact double literals.
    import numpy as np

    np.testing.assert_allclose(row["action_chunk"], np.array([[0.0] * 7, [0.1] * 7], dtype=np.float32))


def test_resume_skips_completed_keys(tmp_path):
    store = EpisodeStore(str(tmp_path / "local"), remote_root=None, sync_every_n_episodes=100)
    key = episode_key("demo_replay_relative", "libero_10", "libero_10:0", 0)
    assert not store.is_done(key)
    store.write_episode(key, _dummy_record())
    assert store.is_done(key)

    other_key = episode_key("demo_replay_relative", "libero_10", "libero_10:0", 1)
    assert not store.is_done(other_key)


def test_resume_checks_remote_not_local_when_configured(tmp_path):
    """Local disk does not survive a Colab session restart -- only Drive does.
    Resume must check the remote/persistent store, not local staging."""
    local = tmp_path / "local"
    remote = tmp_path / "remote"
    store = EpisodeStore(str(local), remote_root=str(remote), sync_every_n_episodes=100)
    key = episode_key("demo_replay_relative", "libero_10", "libero_10:0", 0)

    store.write_episode(key, _dummy_record())
    # Not yet synced (sync_every_n_episodes=100) -> should NOT be considered done.
    assert not store.is_done(key)

    store.sync_to_remote()
    assert store.is_done(key)


def test_batched_sync_does_not_touch_remote_until_threshold(tmp_path):
    local = tmp_path / "local"
    remote = tmp_path / "remote"
    store = EpisodeStore(str(local), remote_root=str(remote), sync_every_n_episodes=3)

    for i in range(2):
        store.write_episode(
            episode_key("demo_replay_relative", "libero_10", "libero_10:0", i), _dummy_record()
        )
    assert list(remote.glob("episodes/*.parquet")) == []

    store.write_episode(
        episode_key("demo_replay_relative", "libero_10", "libero_10:0", 2), _dummy_record()
    )
    assert len(list(remote.glob("episodes/*.parquet"))) == 3


def test_partial_results_still_readable(tmp_path):
    """A run that dies mid-way must still yield a report from whatever completed."""
    store = EpisodeStore(str(tmp_path / "local"), remote_root=None, sync_every_n_episodes=1)
    store.write_episode(
        episode_key("demo_replay_relative", "libero_10", "libero_10:0", 0), _dummy_record()
    )
    # Simulate interruption: no more episodes written.
    table = store.read_all()
    assert table.num_rows == 1
