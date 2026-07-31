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
