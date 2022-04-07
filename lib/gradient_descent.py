from __future__ import annotations

from numpy import dot


class gradientDescent:
    def get_gradient_descent(
        self,
        x: list[list[float]],
        h: list[float],
        y: list[int],
    ) -> list[float]:
        gd = dot(x.T, (h - y)) / y.shape[0]
        return gd
