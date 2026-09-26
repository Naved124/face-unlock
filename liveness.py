"""Blink detection, reusable by any script or the unlock service."""
from dataclasses import dataclass

import numpy as np

EYE_A = list(range(33, 43))
EYE_B = list(range(87, 97))
MOUTH = list(range(52, 72))

CLOSED_T = 0.09
OPEN_T = 0.11
MIN_BLINK_SECONDS = 0.10
MAX_BLINK_SECONDS = 0.8
MAX_SCALE_CHANGE = 0.10
MAX_MOVE = 0.30


def eye_height(points):
    ys = points[:, 1]
    return ys.max() - ys.min()


def measure(lm):
    """Return (relative openness, eyes-to-mouth distance, face center)."""
    eye_center = lm[EYE_A + EYE_B].mean(axis=0)
    mouth_center = lm[MOUTH].mean(axis=0)
    ref = float(np.linalg.norm(mouth_center - eye_center))
    center = (eye_center + mouth_center) / 2
    if ref <= 0:
        return 0.0, 0.0, center
    h = (eye_height(lm[EYE_A]) + eye_height(lm[EYE_B])) / 2
    return h / ref, ref, center


@dataclass
class BlinkEvent:
    ok: bool
    reason: str       # "ok", "too short", "too long", "face moved"
    started: float    # time the eyes closed
    duration: float
    scale: float
    move: float


class BlinkDetector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.closed = False
        self.closed_at = 0.0
        self.last_open_ref = None
        self.last_open_center = None
        self.max_scale = 0.0
        self.max_move = 0.0
        self.rel = 0.0

    def update(self, lm, now):
        """Feed one frame. Returns a BlinkEvent when the eyes reopen, else None."""
        rel, ref, center = measure(lm)
        self.rel = rel

        if not self.closed:
            if rel < CLOSED_T and self.last_open_ref:
                self.closed = True
                self.closed_at = now
                self.max_scale = 0.0
                self.max_move = 0.0
            else:
                self.last_open_ref = ref
                self.last_open_center = center
                return None

        scale = abs(ref - self.last_open_ref) / self.last_open_ref
        move = float(np.linalg.norm(center - self.last_open_center)) / self.last_open_ref
        self.max_scale = max(self.max_scale, scale)
        self.max_move = max(self.max_move, move)

        if rel <= OPEN_T:
            return None  # still closed

        self.closed = False
        duration = now - self.closed_at
        if duration < MIN_BLINK_SECONDS:
            reason = "too short"
        elif duration > MAX_BLINK_SECONDS:
            reason = "too long"
        elif self.max_scale > MAX_SCALE_CHANGE or self.max_move > MAX_MOVE:
            reason = "face moved"
        else:
            reason = "ok"
        return BlinkEvent(reason == "ok", reason, self.closed_at, duration,
                          self.max_scale, self.max_move)