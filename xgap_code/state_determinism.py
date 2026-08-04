"""Pure numeric comparison utilities for scripts/test_state_restore_determinism.py
-- kept separate from any env/sim access (harness.get_sim_state/restore_sim_state)
so this part is testable without a simulator installed."""

import numpy as np


def compare_state_chunks(state_a, state_b, atol: float = 1e-6) -> dict:
    """state_a/state_b: (5,) [eef_pos_x,y,z, gripper_qpos_0,1] arrays, same
    layout as replay.py's _extract_state. Returns {"max_abs_diff": float,
    "identical": bool}."""
    arr_a = np.asarray(state_a, dtype=np.float64)
    arr_b = np.asarray(state_b, dtype=np.float64)
    diff = np.abs(arr_a - arr_b)
    return {"max_abs_diff": float(diff.max()), "identical": bool(np.allclose(arr_a, arr_b, atol=atol))}


def compare_images(img_a, img_b) -> dict:
    """img_a/img_b: (H,W,3) uint8 arrays from env.render(). Returns
    {"max_abs_diff": int, "pct_pixels_differing": float, "identical": bool}
    -- exact pixel equality is checked but not assumed required for the
    overall determinism verdict (rendering can have tiny nondeterminism
    even given identical physics state); reported as a diagnostic number,
    not a hard pass/fail gate on its own."""
    arr_a = np.asarray(img_a, dtype=np.int16)
    arr_b = np.asarray(img_b, dtype=np.int16)
    diff = np.abs(arr_a - arr_b)
    pct_differing = float((diff.max(axis=-1) > 0).mean() * 100.0)
    return {
        "max_abs_diff": int(diff.max()),
        "pct_pixels_differing": pct_differing,
        "identical": bool(diff.max() == 0),
    }
