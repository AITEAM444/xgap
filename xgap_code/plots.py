"""Plotting utilities for comparing a policy rollout's actions against the demo
baseline. Built now (step 2) even though there's no policy rollout data yet
(that's step 3), so the diagnostic tool exists before we have anything to
point it at -- and so it can be exercised now against real demo data (see
scripts/analyze_demo_actions.py's output).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display available in Colab/CI
import matplotlib.pyplot as plt
import numpy as np


def plot_gripper_action_histogram(
    demo_gripper_actions,
    policy_gripper_actions,
    save_path: str,
    title: str = "action[:,6] (gripper) -- demo vs policy",
) -> None:
    """Overlaid histogram of the gripper action dimension.

    The demo distribution is discrete in {-1, +1} (measured from real
    HuggingFaceVLA/libero data -- see outputs/demo_action_stats/). If the
    policy's mass clusters near 0 instead of at the two extremes, that's a
    normalization bug on a dimension that should never be continuous (e.g. a
    MEAN_STD normalizer applied to a naturally bimodal {-1,+1} signal, or an
    unnormalize step using the wrong stats) -- a different failure mode from
    H3's sign-flip question, and visually distinguishable from it: a sign flip
    still shows two spikes (just swapped), a normalization bug shows one blob
    in the middle.
    """
    demo = np.asarray(demo_gripper_actions, dtype=np.float32).ravel()
    policy = np.asarray(policy_gripper_actions, dtype=np.float32).ravel()

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(-1.05, 1.05, 43)
    ax.hist(demo, bins=bins, alpha=0.6, label=f"demo (n={len(demo)})", density=True)
    ax.hist(policy, bins=bins, alpha=0.6, label=f"policy (n={len(policy)})", density=True)
    ax.set_xlabel("action[:, 6]")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_episode_trajectory(
    state_chunk,
    action_chunk,
    save_path: str,
    *,
    compare_state_chunk=None,
    compare_action_chunk=None,
    compare_label: str = "comparison",
    title: str = "episode trajectory",
) -> None:
    """eef position (x,y,z), gripper qpos, and action[:,6] over time for one episode.

    `state_chunk` is (T,5) = [eef_pos_x,y,z, gripper_qpos_0,1] (logging_schema.py's
    state_chunk field -- NOT the policy-facing 8D observation.state). `action_chunk` is
    (T,7). Pass `compare_state_chunk`/`compare_action_chunk` (e.g. a demo episode's) to
    overlay a second trajectory on the same axes -- there's no policy rollout yet to
    compare against (step 3), so this is usable single-series right now and extends
    directly to policy-vs-demo overlays once there is one.
    """
    state = np.asarray(state_chunk, dtype=np.float32)
    action = np.asarray(action_chunk, dtype=np.float32)
    cmp_state = np.asarray(compare_state_chunk, dtype=np.float32) if compare_state_chunk is not None else None
    cmp_action = np.asarray(compare_action_chunk, dtype=np.float32) if compare_action_chunk is not None else None

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    ax = axes[0]
    for i, dim in enumerate("xyz"):
        ax.plot(state[:, i], label=f"eef_{dim}")
    if cmp_state is not None:
        for i, dim in enumerate("xyz"):
            ax.plot(cmp_state[:, i], linestyle="--", alpha=0.6, label=f"{compare_label} eef_{dim}")
    ax.set_ylabel("eef position")
    ax.legend(fontsize=7, ncol=3)

    ax = axes[1]
    ax.plot(state[:, 3], label="gripper_qpos_0")
    ax.plot(state[:, 4], label="gripper_qpos_1")
    if cmp_state is not None:
        ax.plot(cmp_state[:, 3], linestyle="--", alpha=0.6, label=f"{compare_label} gripper_qpos_0")
        ax.plot(cmp_state[:, 4], linestyle="--", alpha=0.6, label=f"{compare_label} gripper_qpos_1")
    ax.set_ylabel("gripper qpos")
    ax.legend(fontsize=7)

    ax = axes[2]
    ax.plot(action[:, 6], label="action[:,6] (gripper cmd)")
    if cmp_action is not None:
        ax.plot(cmp_action[:, 6], linestyle="--", alpha=0.6, label=f"{compare_label} action[:,6]")
    ax.set_ylabel("gripper action")
    ax.set_xlabel("step")
    ax.legend(fontsize=7)

    fig.suptitle(title)
    fig.tight_layout()

    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
