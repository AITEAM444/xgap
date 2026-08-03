#!/usr/bin/env python
"""Diagnostic: replay one fixed demo against ONE specific init_state and save a
video + trajectory plot (overlaid against the demo's own recorded action/state),
for visually confirming a candidate from scripts/sweep_init_states.py's
successful_init_states list.

Deliberately separate from sweep_init_states.py rather than adding a video
option there: that script's resume logic keys on "does this episode's result
file already exist", so turning video on for an already-completed index would
silently skip re-running it and never produce a video. This script always
re-runs regardless of what the sweep already recorded.

    python scripts/render_init_state_video.py --config configs/sweep_init_states.yaml --init-state-index 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xgap_code.config import InitStateSweepConfig  # noqa: E402
from xgap_code.dataset_io import load_task_demo_episodes  # noqa: E402
from xgap_code.harness import get_libero_task_language, make_real_libero_env  # noqa: E402
from xgap_code.plots import plot_episode_trajectory  # noqa: E402
from xgap_code.replay import replay_episode  # noqa: E402


def run(cfg: InitStateSweepConfig, init_state_index: int, out_dir: Path) -> dict:
    task_label = f"{cfg.task_suite}:{cfg.task_id}"
    task_language = get_libero_task_language(cfg.task_suite, cfg.task_id)
    demo_episodes = load_task_demo_episodes(
        cfg.dataset_repo_id, task_name=task_language, max_episodes=cfg.demo_within_task_index + 1
    )
    demo = demo_episodes[cfg.demo_within_task_index]

    env = make_real_libero_env(
        task_suite_name=cfg.task_suite,
        task_id=cfg.task_id,
        demo_episode_index_within_task=init_state_index,
        control_mode=cfg.control_mode,
        control_freq=cfg.control_freq,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / f"init_state_{init_state_index}.mp4"
    record = replay_episode(
        env,
        demo.actions,
        task_id=task_label,
        task_suite=cfg.task_suite,
        episode_seed=init_state_index,
        environment_seed=init_state_index,
        condition="init_state_render",
        control_mode=cfg.control_mode,
        control_freq=cfg.control_freq,
        checkpoint_name=cfg.checkpoint_name,
        checkpoint_hash=cfg.checkpoint_hash,
        git_commit="n/a",
        config_file="n/a",
        video_path=video_path,
        video_sample_every_n_steps=1,
    )
    env.close()

    # demo.states is (T,8) = eef_pos(3) + axis-angle(3) + gripper_qpos(2) -- the
    # policy-facing layout (dataset_io.py's DemoEpisode). replay's state_chunk is
    # (T,5) = eef_pos(3) + gripper_qpos(2) (replay.py's _extract_state) -- drop the
    # axis-angle columns so the two line up for the overlay plot.
    demo_state_5d = demo.states[:, [0, 1, 2, 6, 7]]

    plot_path = out_dir / f"init_state_{init_state_index}_vs_demo.png"
    plot_episode_trajectory(
        record.state_chunk,
        record.action_chunk,
        str(plot_path),
        compare_state_chunk=demo_state_5d,
        compare_action_chunk=demo.actions,
        compare_label="demo",
        title=f"{task_label} init_state={init_state_index} vs recorded demo",
    )

    return {
        "task": task_label,
        "init_state_index": init_state_index,
        "success": record.episode_success,
        "rollout_length": record.rollout_length,
        "video_path": str(video_path),
        "plot_path": str(plot_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--init-state-index", type=int, required=True)
    parser.add_argument("--out-dir", default=None, help="Defaults to <local_output_root>/render")
    args = parser.parse_args()

    cfg = InitStateSweepConfig.from_yaml(args.config)
    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg.local_output_root) / "render"
    result = run(cfg, args.init_state_index, out_dir)
    print(result)


if __name__ == "__main__":
    main()
