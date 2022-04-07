from __future__ import annotations

from numpy import log


class Loss:
    def get_loss(self, h: list[float], y: list[int]) -> float:
        loss = (-y * log(h) - (1 - y) * log(1 - h)).mean()
        return loss
