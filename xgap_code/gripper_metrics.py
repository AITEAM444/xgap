"""Gripper-command timing analysis, shared between demo replay and (later)
policy rollout instrumentation.

The H3 baseline measured from real HuggingFaceVLA/libero demos this session
(see outputs/demo_action_stats/) shows action[:,6] is discrete in {-1 (open),
+1 (close)}, and the physical gripper takes roughly 15-20 simulation steps to
visibly close (finger_gap collapse) after a +1 command -- actuator lag, not
instantaneous. A close command held for fewer consecutive steps than that
cannot physically result in a grasp, REGARDLESS of sign convention -- this is
a distinct, checkable failure mode from H3's sign-flip question: a rollout can
have the sign right and still never hold "close" long enough to grasp.
"""

from __future__ import annotations

import numpy as np

# Lower bound of the actuation lag observed in outputs/demo_action_stats/ (measured
# from 5 real demo episodes across libero_10/libero_spatial). A rollout whose longest
# continuous close-command run is below this cannot physically have grasped.
DEMO_MIN_ACTUATION_LAG_STEPS = 15


def longest_close_run(actions, gripper_dim: int = 6, close_threshold: float = 0.0) -> int:
    """Longest run of consecutive steps with action[:, gripper_dim] > close_threshold
    (i.e. commanding "close", per the -1=open / +1=close convention measured from
    demos). Returns 0 if the gripper is never commanded closed. `actions` may be a
    (T, action_dim) array or an equivalent nested list."""
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return 0
    is_close = arr[:, gripper_dim] > close_threshold
    best = current = 0
    for v in is_close:
        current = current + 1 if v else 0
        best = max(best, current)
    return int(best)
