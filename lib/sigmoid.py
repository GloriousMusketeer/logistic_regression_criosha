from __future__ import annotations

from numpy import array
from numpy import exp


class Sigmoid:
    def sigmoid_function(self, x: list[float]) -> list[float]:
        if x >= 0:
            sig = 1 / (1 + exp(-x))
        else:
            """
            si x es menor que cero, entonces z será pequeño,
            el denominador no puede ser cero porque es 1+z. 
            """
            sig = exp(x) / (1 + exp(x))
        return sig

    def sigmoid(self, x):
        sig = array([self.sigmoid_function(value) for value in x])
        return sig

