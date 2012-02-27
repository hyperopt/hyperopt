import numpy as np
from hyperopt.utils import fast_isin


def test_fast_isin():
    Y = np.random.randint(0, 10000, size=(100, ))
    X = np.arange(10000)
    Z = fast_isin(X, Y)
    D = np.unique(Y)
    D.sort()
    T1 = (X[Z] == D).all()

    X = np.array(range(10000) + range(10000))
    Z = fast_isin(X, Y)
    T2 = (X[Z] == np.append(D, D.copy())).all()

    X = np.random.randint(0, 100, size = (40, ))
    X.sort()
    Y = np.random.randint(0, 100, size = (60, ))
    Y.sort()

    XinY = np.array([ind for ind in range(len(X)) if X[ind] in Y])
    YinX = np.array([ind for ind in range(len(Y)) if Y[ind] in X])


    T3 = (fast_isin(X, Y).nonzero()[0] == XinY).all()
    T4 = (fast_isin(Y, X).nonzero()[0] == YinX).all()


    assert T1 & T2 & T3 & T4
