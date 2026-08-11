"""Unit tests for Gate 2's pure diversity metrics -- no env/policy involved."""

from __future__ import annotations

import numpy as np

from xgap_code.diversity_metrics import (
    endpoint_variance,
    gripper_channel_distribution,
    mean_pairwise_l2,
    per_dimension_std,
)


def test_per_dimension_std_identical_candidates_is_zero():
    chunks = np.tile(np.array([[1.0, 2.0, 3.0]]), (10, 1))
    stds = per_dimension_std(chunks)
    assert all(s == 0.0 for s in stds)


def test_per_dimension_std_detects_spread():
    chunks = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    stds = per_dimension_std(chunks)
    assert stds[0] > 0.0
    assert stds[1] == 0.0


def test_per_dimension_std_handles_chunked_input():
    # (N_candidates=2, T=3, action_dim=2)
    chunks = np.array([
        [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
    ])
    stds = per_dimension_std(chunks)
    assert stds[0] == 0.5  # half the candidates at 0, half at 1
    assert stds[1] == 0.0


def test_mean_pairwise_l2_identical_is_zero():
    chunks = np.tile(np.array([[1.0, 2.0]]), (5, 1))
    assert mean_pairwise_l2(chunks) == 0.0


def test_mean_pairwise_l2_single_candidate_is_zero():
    chunks = np.array([[1.0, 2.0]])
    assert mean_pairwise_l2(chunks) == 0.0


def test_mean_pairwise_l2_matches_manual_calc_for_two_points():
    chunks = np.array([[0.0, 0.0], [3.0, 4.0]])  # distance = 5
    assert mean_pairwise_l2(chunks) == 5.0


def test_gripper_channel_distribution_all_closed():
    # (N=4, T=1, action_dim=7), gripper dim (index 6) always 1.0 (close)
    chunks = np.zeros((4, 1, 7))
    chunks[:, :, 6] = 1.0
    stats = gripper_channel_distribution(chunks)
    assert stats["mean"] == 1.0
    assert stats["std"] == 0.0
    assert stats["frac_commanding_close"] == 1.0


def test_gripper_channel_distribution_mixed():
    chunks = np.zeros((4, 1, 7))
    chunks[0, 0, 6] = 1.0
    chunks[1, 0, 6] = 1.0
    chunks[2, 0, 6] = -1.0
    chunks[3, 0, 6] = -1.0
    stats = gripper_channel_distribution(chunks)
    assert stats["frac_commanding_close"] == 0.5
    assert stats["mean"] == 0.0


def test_endpoint_variance_identical_endpoints_is_zero():
    endpoints = np.tile(np.array([[0.1, 0.2, 0.3]]), (10, 1))
    result = endpoint_variance(endpoints)
    # Tiny floating-point roundoff from mean-of-identical-values, not real
    # spread -- assert near-zero rather than exact 0.0 (0.1 isn't exactly
    # representable in binary float64).
    assert all(v < 1e-20 for v in result["per_dim_variance"])
    assert result["total_variance"] < 1e-20
    assert result["mean_pairwise_l2"] < 1e-10


def test_endpoint_variance_detects_spread():
    endpoints = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    result = endpoint_variance(endpoints)
    assert result["per_dim_variance"][0] > 0.0
    assert result["per_dim_variance"][1] == 0.0
    assert result["mean_pairwise_l2"] > 0.0
