"""Per-stage wall-clock timing, for the cost breakdown required by the project spec:
policy inference / rendering / physics step / reset."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class StageTimer:
    totals: dict[str, float] = field(default_factory=lambda: {
        "inference": 0.0, "render": 0.0, "physics": 0.0, "reset": 0.0,
    })

    @contextmanager
    def stage(self, name: str):
        if name not in self.totals:
            raise ValueError(f"Unknown timing stage '{name}', expected one of {list(self.totals)}")
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.totals[name] += time.perf_counter() - t0

    def as_fields(self) -> dict[str, float]:
        return {
            "time_inference_s": self.totals["inference"],
            "time_render_s": self.totals["render"],
            "time_physics_s": self.totals["physics"],
            "time_reset_s": self.totals["reset"],
        }
