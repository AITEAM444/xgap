#!/usr/bin/env python
"""Diagnostic: replay ONE demo's recorded action sequence against EVERY candidate
init_state for its task -- tests directly whether `within_task_index`
(xgap_code/dataset_io.py) picks the init_state LIBERO actually recorded this demo
from, instead of continuing to infer it indirectly from position/orientation plots.

If any init_state index succeeds: init_state indexing is confirmed as the bug, and
this sweep hands you the correct index directly. If none succeed: init_state
indexing is exonerated, and orientation (not yet logged in state_chunk) becomes
the next thing to check. See README "leading hypothesis: orientation".

    python scripts/sweep_init_states.py --config configs/sweep_init_states.yaml

No --mock mode: this only makes sense against the real simulator (MockLiberoEnv
has no concept of "N init_states"). Requires lerobot[libero] installed
(Linux/Colab only).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xgap_code.config import InitStateSweepConfig  # noqa: E402
from xgap_code.dataset_io import load_task_demo_episodes  # noqa: E402
from xgap_code.harness import get_libero_task_language, get_num_init_states, make_real_libero_env  # noqa: E402
from xgap_code.logging_schema import EpisodeStore, episode_key  # noqa: E402
from xgap_code.replay import replay_episode  # noqa: E402


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def run(cfg: InitStateSweepConfig, config_file: str) -> dict:
    store = EpisodeStore(cfg.local_output_root, cfg.remote_output_root, cfg.sync_every_n_episodes)
    git_commit = _git_commit()

    task_label = f"{cfg.task_suite}:{cfg.task_id}"
    task_language = get_libero_task_language(cfg.task_suite, cfg.task_id)
    demo_episodes = load_task_demo_episodes(
        cfg.dataset_repo_id, task_name=task_language, max_episodes=cfg.demo_within_task_index + 1
    )
    demo = demo_episodes[cfg.demo_within_task_index]
    print(
        f"[sweep] task='{task_language}' ({task_label}), using demo "
        f"within_task_index={cfg.demo_within_task_index} (dataset episode_index={demo.episode_index}), "
        f"{len(demo.actions)} steps"
    )

    n_init_states = get_num_init_states(cfg.task_suite, cfg.task_id)
    print(f"[sweep] {n_init_states} candidate init_states for this task")

    condition = "init_state_sweep"
    results: list[dict] = []
    for init_idx in range(n_init_states):
        key = episode_key(condition, cfg.task_suite, task_label, init_idx)

        if cfg.resume and store.is_done(key):
            import pyarrow.parquet as pq

            row = pq.read_table(store.persistent_dir() / f"{key}.parquet").to_pylist()[0]
            results.append({"init_state_index": init_idx, "success": bool(row["episode_success"])})
            continue

        env = make_real_libero_env(
            task_suite_name=cfg.task_suite,
            task_id=cfg.task_id,
            demo_episode_index_within_task=init_idx,
            control_mode=cfg.control_mode,
            control_freq=cfg.control_freq,
        )
        record = replay_episode(
            env,
            demo.actions,
            task_id=task_label,
            task_suite=cfg.task_suite,
            episode_seed=init_idx,
            environment_seed=init_idx,
            condition=condition,
            control_mode=cfg.control_mode,
            control_freq=cfg.control_freq,
            checkpoint_name=cfg.checkpoint_name,
            checkpoint_hash=cfg.checkpoint_hash,
            git_commit=git_commit,
            config_file=config_file,
        )
        env.close()
        store.write_episode(key, record)
        results.append({"init_state_index": init_idx, "success": record.episode_success})
        print(f"  init_state {init_idx}: success={record.episode_success}")
        if record.episode_success:
            print(f"[sweep] *** init_state {init_idx} SUCCEEDS with this demo's actions ***")

    store.sync_to_remote()

    successes = [r["init_state_index"] for r in results if r["success"]]
    summary = {
        "task": task_label,
        "task_language": task_language,
        "demo_episode_index": demo.episode_index,
        "n_init_states_tested": len(results),
        "successful_init_states": successes,
        "verdict": (
            f"init_state indexing bug confirmed -- correct index is {successes}"
            if successes
            else "init_state indexing exonerated -- none of the candidate init_states reproduce "
            "success with this demo's actions. Move to orientation (see README)."
        ),
    }
    out_path = store.persistent_dir().parent / "sweep_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = InitStateSweepConfig.from_yaml(args.config)
    summary = run(cfg, config_file=args.config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
