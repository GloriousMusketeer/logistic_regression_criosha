from __future__ import annotations

from numpy import dot
from numpy import sum


class gradientDescent:
    def get_gradient_descent(
        self,
        x: list[list[float]],
        h: list[float],
        y: list[int],
    ) -> list[float]:
        dw = dot(x.T, (h - y)) / y.shape[0]
        db = sum((h - y)) / y.shape[0]

        return (dw, db)
