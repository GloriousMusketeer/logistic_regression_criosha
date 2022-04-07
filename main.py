from __future__ import annotations

from logistic import logisticFunction

from sklearn.datasets import load_iris


lf = logisticFunction()


def main() -> int:
    li = load_iris()
    x = li.data[:, :2]
    y = (li.target != 0) * 1
    x_index = 0
    y_index = 1
    x_label = li.feature_names[x_index]
    y_label = li.feature_names[y_index]
    lf.logistic(x, y, x_label, y_label)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
