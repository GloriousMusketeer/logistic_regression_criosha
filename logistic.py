from __future__ import annotations

from lib.sigmoid import Sigmoid
from lib.loss import Loss
from lib.gradient_descent import gradientDescent

from numpy import concatenate
from numpy import dot
from numpy import ones
from numpy import zeros


# var
gd = gradientDescent()
lss = Loss()
smd = Sigmoid()

# global
weight: list[float] = []


class logisticFunction:
    def logistic(self, x: list[list[float]], y: list[int]) -> None:
        intercept: list[list[float]] = ones((x.shape[0], 1))
        x_t: list[list[float]] = concatenate((intercept, x), axis=1)
        weight: list[float] = zeros(x_t.shape[1])
        y_t: list[int] = y
        lr = 0.01
        iterations = 10000
        umbral = 0.5
        self.fit(x_t, weight, y_t, lr, iterations)
        pred = self.predict(x, umbral, intercept, weight)
        print((pred == y_t).mean())

    def fit(
        self,
        x_t: list[list[float]],
        weight: list[float],
        y_t: list[int],
        lr: float,
        iterations: int,
    ) -> str:
        for _ in range(iterations):
            z = dot(x_t, weight)
            h = smd.sigmoid(z)
            _ = lss.get_loss(h, y_t)
            dw = gd.get_gradient_descent(x_t, h, y_t)
            weight -= lr * dw
        return print("proceso de ajustes sin problemas")

    def predict(
        self,
        x: list[list[float]],
        umbral: float,
        intercept: list[list[float]],
        weight: list[float]
    ) -> None:
        x_new = concatenate((intercept, x), axis=1)
        z = dot(x_new, weight)
        result = smd.sigmoid(z)
        result = result >= umbral
        y_pred = zeros(result.shape[0])
        for i in range(len(y_pred)):
            if result[i] == True:
                y_pred[i] = 1
            else:
                continue
        return y_pred
