"""Unit tests for the pure numeric comparison functions used by
scripts/test_state_restore_determinism.py -- no simulator involved, matching
this project's established pattern of testing logic separately from env/sim
access (see xgap_code/harness.py's get_sim_state/restore_sim_state, which
this deliberately does not touch)."""

from __future__ import annotations

import numpy as np

from xgap_code.state_determinism import compare_images, compare_state_chunks


def test_compare_state_chunks_identical():
    state = [0.1, 0.2, 0.3, 0.04, -0.04]
    result = compare_state_chunks(state, state)
    assert result["identical"] is True
    assert result["max_abs_diff"] == 0.0


def test_compare_state_chunks_different():
    a = [0.1, 0.2, 0.3, 0.04, -0.04]
    b = [0.1, 0.2, 0.35, 0.04, -0.04]
    result = compare_state_chunks(a, b)
    assert result["identical"] is False
    assert result["max_abs_diff"] > 0.0


def test_compare_state_chunks_within_tolerance():
    a = [0.1, 0.2, 0.3, 0.04, -0.04]
    b = [0.1, 0.2, 0.3 + 1e-9, 0.04, -0.04]
    result = compare_state_chunks(a, b)
    assert result["identical"] is True


def test_compare_images_identical():
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img[2, 2] = [255, 0, 0]
    result = compare_images(img, img)
    assert result["identical"] is True
    assert result["max_abs_diff"] == 0
    assert result["pct_pixels_differing"] == 0.0


def test_compare_images_different():
    img_a = np.zeros((8, 8, 3), dtype=np.uint8)
    img_b = np.zeros((8, 8, 3), dtype=np.uint8)
    img_b[0, 0] = [255, 255, 255]  # 1 of 64 pixels differs
    result = compare_images(img_a, img_b)
    assert result["identical"] is False
    assert result["max_abs_diff"] == 255
    assert result["pct_pixels_differing"] == 100.0 / 64
