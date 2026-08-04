#!/usr/bin/env python
"""Gate-1/N=1 harness-parity check: run the real SmolVLA policy through
xgap's own harness (harness.make_real_libero_env + policy_rollout.py), using
STANDARD init_state order -- and compare the resulting success rate against
the official `lerobot-eval` CLI's own number on the same task/episode count.
See README "libero_spatial로 변경한다" for why libero_spatial and why this
comparison matters before scaling up to the full Gate-1 measurement.

    python scripts/run_policy_rollout.py --config configs/policy_rollout_libero_spatial_smoke.yaml
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xgap_code.config import PolicyRolloutConfig  # noqa: E402
from xgap_code.harness import get_libero_task_language, make_real_libero_env  # noqa: E402
from xgap_code.logging_schema import EpisodeStore, episode_key  # noqa: E402
from xgap_code.policy_rollout import build_policy_and_processors, rollout_policy_episode  # noqa: E402


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def run(cfg: PolicyRolloutConfig, config_file: str) -> dict:
    store = EpisodeStore(cfg.local_output_root, cfg.remote_output_root, cfg.sync_every_n_episodes)
    git_commit = _git_commit()

    condition = "policy_rollout"
    results: list[dict] = []

    for task_id in cfg.task_ids:
        task_label = f"{cfg.task_suite}:{task_id}"
        task_description = get_libero_task_language(cfg.task_suite, task_id)
        print(f"[rollout] task='{task_description}' ({task_label})")

        policy, preprocessor, postprocessor, env_preprocessor, _env_postprocessor = build_policy_and_processors(
            checkpoint_path=cfg.checkpoint_path,
            task_suite_name=cfg.task_suite,
            task_id=task_id,
            control_freq=cfg.control_freq,
        )

        for episode_seed in range(cfg.episodes_per_task):
            key = episode_key(condition, cfg.task_suite, task_label, episode_seed)

            if cfg.resume and store.is_done(key):
                import pyarrow.parquet as pq

                row = pq.read_table(store.persistent_dir() / f"{key}.parquet").to_pylist()[0]
                results.append({"task_id": task_id, "episode_seed": episode_seed, "success": bool(row["episode_success"])})
                print(f"  task={task_id} episode={episode_seed}: success={bool(row['episode_success'])} (resumed)")
                continue

            env = make_real_libero_env(
                task_suite_name=cfg.task_suite,
                task_id=task_id,
                demo_episode_index_within_task=episode_seed,  # STANDARD order -- see harness.py docstring
                control_mode=cfg.control_mode,
                control_freq=cfg.control_freq,
            )
            record = rollout_policy_episode(
                env,
                policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                env_preprocessor=env_preprocessor,
                task_description=task_description,
                episode_seed=episode_seed,
                max_steps=cfg.max_steps,
                task_id=task_label,
                task_suite=cfg.task_suite,
                condition=condition,
                control_mode=cfg.control_mode,
                control_freq=cfg.control_freq,
                checkpoint_name=cfg.checkpoint_path,
                checkpoint_hash=cfg.checkpoint_hash,
                git_commit=git_commit,
                config_file=config_file,
            )
            env.close()
            store.write_episode(key, record)
            results.append({"task_id": task_id, "episode_seed": episode_seed, "success": record.episode_success})
            print(
                f"  task={task_id} episode={episode_seed}: success={record.episode_success} "
                f"(rollout_length={record.rollout_length}, longest_close_run={record.longest_close_run_steps})"
            )

    store.sync_to_remote()

    n = len(results)
    n_success = sum(1 for r in results if r["success"])
    summary = {
        "task_suite": cfg.task_suite,
        "task_ids": cfg.task_ids,
        "n_episodes": n,
        "n_success": n_success,
        "pc_success": (n_success / n * 100.0) if n else float("nan"),
        "results": results,
    }
    out_path = store.persistent_dir().parent / "rollout_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = PolicyRolloutConfig.from_yaml(args.config)
    summary = run(cfg, config_file=args.config)
    print(f"\n[rollout] {summary['n_success']}/{summary['n_episodes']} = {summary['pc_success']:.1f}%")


if __name__ == "__main__":
    main()
