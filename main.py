from __future__ import annotations

from logistic import logisticFunction

from sklearn.datasets import load_iris


lf = logisticFunction()


def main() -> int:
    li = load_iris()
    x = li.data[:, :2]
    y = (li.target != 0) * 1
    lf.logistic(x, y)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
