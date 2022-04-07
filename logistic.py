from __future__ import annotations

from lib.gradient_descent import gradientDescent
from lib.loss import Loss
from lib.sigmoid import Sigmoid

from matplotlib import pyplot as plt
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
    def logistic(
        self,
        x: list[list[float]],
        y: list[int],
        xlabel: str,
        ylabel: str,
    ) -> None:
        intercept: list[list[float]] = ones((x.shape[0], 1))
        x_t: list[list[float]] = concatenate((intercept, x), axis=1)
        (x_min, x_max) = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
        weight: list[float] = zeros(x_t.shape[1])
        y_t: list[int] = y
        (y_min, y_max) = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5
        lr = 0.01
        iterations = 10000
        umbral = 0.5
        self.fit(x_t, weight, y_t, lr, iterations)
        pred = self.predict(x, umbral, intercept, weight)
        print((pred == y_t).mean())
        plt.figure(2, figsize=(8, 6))
        plt.clf()
        plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Set1, edgecolor="k")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        plt.tight_layout()
        plt.show()

    def fit(
        self,
        x_t: list[list[float]],
        weight: list[float],
        y_t: list[int],
        lr: float,
        iterations: int,
    ) -> str:
        # agregamos bias para la función z.
        # z = wx+b
        bias = 0
        # coloco en los for (_) por que no voy a necesitar la iteración.
        # pero si voy a necesitar que que recorra toda las iteraciones
        for _ in range(iterations):
            z = dot(x_t, weight) + bias
            h = smd.sigmoid(z)
            # lo mismo pasa con la salida de loss, no voy a usar la salida.
            # pero si necesito que haga la función para entrenar
            _ = lss.get_loss(h, y_t)
            (dw, db) = gd.get_gradient_descent(x_t, h, y_t)
            weight -= lr * dw
            bias -= lr * db

        return print("proceso de ajustes sin problemas")

    def predict(
        self,
        x: list[list[float]],
        umbral: float,
        intercept: list[list[float]],
        weight: list[float],
    ) -> list[float]:
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
