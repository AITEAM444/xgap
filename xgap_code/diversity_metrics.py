"""Gate 2 (candidate diversity) metrics -- pure numeric computation, no
env/policy access, so testable without a simulator (see
tests/test_diversity_metrics.py). Consumed by scripts/run_gate2_diversity.py.

Diversity here must come ONLY from the policy's own intrinsic
stochasticity (e.g. flow-matching sampling noise) -- these functions don't
know or care where the candidates came from, they just measure how
different a set of them are.
"""

from __future__ import annotations

import numpy as np


def per_dimension_std(action_chunks: np.ndarray) -> list[float]:
    """action_chunks: (N_candidates, T, action_dim) or (N_candidates, action_dim)
    for single-step candidates. Returns per-action-dimension std, computed
    across candidates (and across T if chunked) -- one number per action dim."""
    arr = np.asarray(action_chunks, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr.reshape(-1, arr.shape[-1])
    return arr.std(axis=0).tolist()


def mean_pairwise_l2(action_chunks: np.ndarray) -> float:
    """Mean L2 distance between every pair of candidates (flattening each
    candidate's full chunk into one vector first, so this captures
    whole-trajectory difference, not just a single step's)."""
    arr = np.asarray(action_chunks, dtype=np.float64)
    n = arr.shape[0]
    flat = arr.reshape(n, -1)
    if n < 2:
        return 0.0
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(float(np.linalg.norm(flat[i] - flat[j])))
    return float(np.mean(dists))


def gripper_channel_distribution(action_chunks: np.ndarray, gripper_dim: int = 6) -> dict:
    """action_chunks: (N_candidates, T, action_dim) or (N_candidates, action_dim).
    Returns summary stats for action[..., gripper_dim] across all candidates
    (and steps, if chunked) -- mean/std/min/max, plus fraction of values
    commanding "close" (> 0, per the -1=open/+1=close convention measured
    from real demos -- see gripper_metrics.py)."""
    arr = np.asarray(action_chunks, dtype=np.float64)
    gripper_vals = arr[..., gripper_dim].ravel()
    return {
        "mean": float(gripper_vals.mean()),
        "std": float(gripper_vals.std()),
        "min": float(gripper_vals.min()),
        "max": float(gripper_vals.max()),
        "frac_commanding_close": float((gripper_vals > 0).mean()),
    }


def endpoint_variance(endpoints: np.ndarray) -> dict:
    """endpoints: (N_candidates, D) -- e.g. final eef position (D=3) or full
    state (D=5) reached after actually EXECUTING each candidate. Returns
    per-dimension variance and the mean pairwise L2 distance between
    endpoints (same "how spread out" question as mean_pairwise_l2, but on
    where candidates actually ENDED UP physically, not on the raw action
    values that produced them)."""
    arr = np.asarray(endpoints, dtype=np.float64)
    per_dim_var = arr.var(axis=0).tolist()
    return {
        "per_dim_variance": per_dim_var,
        "total_variance": float(np.sum(per_dim_var)),
        "mean_pairwise_l2": mean_pairwise_l2(arr),
    }
