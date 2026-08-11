#!/usr/bin/env python
"""List real episode_seed -> episode_success for a task's recorded source
episodes (e.g. a completed PolicyRolloutConfig run's own EpisodeStore
output), so real SUCCESS and FAILURE seeds can be picked for
scripts/run_gate2_outcome.py's `source_episode_seeds` -- see README "Gate
2, corrected: outcome must branch from failures too, not only successes"
for why branching only from success episodes was invalid (near-guarantees
the handed-off base policy also succeeds, regardless of candidate choice,
so it never actually tests whether candidate diversity matters). No
guessing at seed numbers -- this reads the real recorded outcome per seed
directly.

    python scripts/list_source_episode_outcomes.py \
        --source-output-root /content/drive/MyDrive/xgap/outputs/policy_rollout_libero_spatial_gate1 \
        --condition policy_rollout --task-suite libero_spatial --task-id 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow.parquet as pq  # noqa: E402

from xgap_code.logging_schema import episode_key  # noqa: E402


def list_episode_outcomes(source_output_root: str, condition: str, task_suite: str, task_id: int) -> list[dict]:
    episodes_dir = Path(source_output_root) / "episodes"
    task_label = f"{task_suite}:{task_id}"
    # episode_key's own sanitization -- reused directly rather than re-implemented,
    # so this always matches exactly what the recording side actually wrote.
    safe_task = episode_key(condition, task_suite, task_label, 0).split("__")[2]

    rows = []
    for path in sorted(episodes_dir.glob(f"{condition}__{task_suite}__{safe_task}__*.parquet")):
        seed_str = path.stem.rsplit("__", 1)[-1]
        if not seed_str.lstrip("-").isdigit():
            continue
        table = pq.read_table(path, columns=["episode_success", "rollout_length"])
        row = table.to_pylist()[0]
        rows.append({"episode_seed": int(seed_str), "episode_success": row["episode_success"],
                      "rollout_length": row["rollout_length"]})
    rows.sort(key=lambda r: r["episode_seed"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-output-root", required=True)
    parser.add_argument("--condition", default="policy_rollout")
    parser.add_argument("--task-suite", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    args = parser.parse_args()

    rows = list_episode_outcomes(args.source_output_root, args.condition, args.task_suite, args.task_id)
    if not rows:
        print(f"No recorded episodes found for task {args.task_id} under {args.source_output_root}")
        return

    successes = [r["episode_seed"] for r in rows if r["episode_success"]]
    failures = [r["episode_seed"] for r in rows if not r["episode_success"]]
    for r in rows:
        print(f"  seed={r['episode_seed']:3d} success={r['episode_success']!s:5} rollout_length={r['rollout_length']}")
    print(f"\ntask {args.task_id}: {len(successes)}/{len(rows)} succeeded")
    print(f"success seeds: {successes}")
    print(f"failure seeds: {failures}")


if __name__ == "__main__":
    main()