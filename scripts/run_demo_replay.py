#!/usr/bin/env python
"""CLI: replay LIBERO demo actions through the eval harness and record results.
This is the answer key for step 2 -- it validates the environment, task
init-state selection, and success detection BEFORE any policy is involved.

    python scripts/run_demo_replay.py --config configs/demo_replay.yaml
    python scripts/run_demo_replay.py --config configs/demo_replay.yaml --mock

`--mock` swaps in xgap_code.harness.MockLiberoEnv and a synthetic demo action
sequence instead of the real LiberoEnv + HuggingFaceVLA/libero dataset, so the
full control flow (resume, incremental save, control_mode decision) can be
exercised without mujoco/libero/lerobot installed. This is how this script was
validated during development -- see README "Local environment check". Real
runs (no --mock) require lerobot[libero] installed (Linux/Colab only).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xgap_code.config import DemoReplayConfig  # noqa: E402
from xgap_code.harness import MockLiberoEnv, get_libero_task_language, make_real_libero_env  # noqa: E402
from xgap_code.logging_schema import EpisodeStore, episode_key  # noqa: E402
from xgap_code.replay import (  # noqa: E402
    decide_env_convention,
    replay_episode,
    summarize_by_control_mode,
    verify_control_mode_wiring,
)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _mock_demo_actions(seed: int, length: int = 50) -> np.ndarray:
    # length must match MockLiberoEnv's default max_steps (50): the mock only ever
    # reports is_success on the step where it terminates (_t >= max_steps), so a
    # shorter action sequence would never trigger success regardless of gripper
    # command -- that's a mock-harness artifact, not a real environment finding.
    """Synthetic stand-in for a demo action sequence: settle, then close the
    gripper near the end -- mirrors the -1(open)->+1(close) pattern found in
    the real HuggingFaceVLA/libero data (see outputs/demo_action_stats/)."""
    rng = np.random.default_rng(seed)
    actions = rng.uniform(-0.1, 0.1, size=(length, 7)).astype(np.float32)
    actions[:, 6] = -1.0
    actions[-5:, 6] = 1.0
    return actions


def _load_existing_row(store: EpisodeStore, key: str) -> dict | None:
    import pyarrow.parquet as pq

    path = store.persistent_dir() / f"{key}.parquet"
    if not path.exists():
        return None
    return pq.read_table(path).to_pylist()[0]


def _write_decision(store: EpisodeStore, decision: dict) -> None:
    path = store.persistent_dir().parent / "control_mode_decision.json"
    path.write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")


def _run_one_episode(
    cfg: DemoReplayConfig, config_file: str, mock: bool, git_commit: str,
    suite: str, task_id, task_label: str, episode_seed: int, control_mode: str,
):
    condition = f"demo_replay_{control_mode}"
    if mock:
        env = MockLiberoEnv(control_mode=control_mode)
        demo_actions = _mock_demo_actions(episode_seed)
    else:
        env = make_real_libero_env(
            task_suite_name=suite,
            task_id=task_id,
            demo_episode_index_within_task=episode_seed,
            control_mode=control_mode,
            control_freq=cfg.control_freq,
        )
        from xgap_code.dataset_io import load_task_demo_episodes

        # task_label ("suite:task_id", e.g. "libero_10:0") is OUR bookkeeping key, not
        # what the dataset's episode metadata is keyed by -- that needs LIBERO's own
        # natural-language task instruction. Conflating the two is exactly the bug that
        # produced "episode filter did not match any episode" on an actual Colab run
        # (see dataset_io.py's module docstring).
        task_language = get_libero_task_language(suite, task_id)
        demo_episodes = load_task_demo_episodes(
            cfg.dataset_repo_id, task_name=task_language, max_episodes=cfg.episodes_per_task
        )
        demo_actions = demo_episodes[episode_seed].actions

    record = replay_episode(
        env,
        demo_actions,
        task_id=task_label,
        task_suite=suite,
        episode_seed=episode_seed,
        environment_seed=episode_seed,
        condition=condition,
        control_mode=control_mode,
        control_freq=cfg.control_freq,
        checkpoint_name=cfg.checkpoint_name,
        checkpoint_hash=cfg.checkpoint_hash,
        git_commit=git_commit,
        config_file=config_file,
    )
    env.close()
    return record


def run(cfg: DemoReplayConfig, config_file: str, mock: bool) -> dict:
    store = EpisodeStore(cfg.local_output_root, cfg.remote_output_root, cfg.sync_every_n_episodes)
    git_commit = _git_commit()
    all_rows: list[dict] = []
    # Populated from the FIRST episode we actually run for each control_mode
    # (across the whole sweep, not per-task) -- one reading is enough to catch a
    # wiring bug, and checking early means we abort before wasting Colab time on
    # a sweep whose control_mode never reached the simulator. Not reset per task:
    # if resume reused a cached row for the first task's episodes, we still want
    # the earliest real reading available.
    actual_mode_by_requested: dict[str, str | None] = {}
    wiring_verified = False

    for suite in cfg.task_suites:
        task_ids = (cfg.task_ids or {}).get(suite) or [0]
        for task_id in task_ids:
            task_label = f"{suite}:{task_id}"
            for episode_seed in range(cfg.episodes_per_task):
                for control_mode in cfg.control_modes:
                    key = episode_key(f"demo_replay_{control_mode}", suite, task_label, episode_seed)

                    if cfg.resume and store.is_done(key):
                        row = _load_existing_row(store, key)
                        if row is not None:
                            all_rows.append(row)
                            actual_mode_by_requested.setdefault(control_mode, row.get("actual_control_mode"))
                        continue

                    record = _run_one_episode(
                        cfg, config_file, mock, git_commit,
                        suite, task_id, task_label, episode_seed, control_mode,
                    )
                    store.write_episode(key, record)
                    all_rows.append(record.to_row())
                    actual_mode_by_requested.setdefault(control_mode, record.actual_control_mode)

                # As soon as we have one reading per requested control_mode, verify
                # wiring BEFORE running anything else -- do not wait until the full
                # sweep finishes and do not look at success rates first.
                if not wiring_verified and len(actual_mode_by_requested) == len(cfg.control_modes):
                    wiring_verified = True
                    abort = verify_control_mode_wiring(actual_mode_by_requested)
                    if abort is not None:
                        store.sync_to_remote()
                        _write_decision(store, abort)
                        return abort

    store.sync_to_remote()
    mode_summary = summarize_by_control_mode(all_rows)
    decision = decide_env_convention(mode_summary)
    _write_decision(store, decision)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mock", action="store_true", help="use MockLiberoEnv (no simulator required)"
    )
    args = parser.parse_args()

    cfg = DemoReplayConfig.from_yaml(args.config)
    decision = run(cfg, config_file=args.config, mock=args.mock)
    print(json.dumps(decision, indent=2, default=str))


if __name__ == "__main__":
    main()
