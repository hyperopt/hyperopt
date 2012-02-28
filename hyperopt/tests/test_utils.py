import numpy as np
from hyperopt.utils import fast_isin
from hyperopt.utils import get_most_recent_inds


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


def test_get_most_recent_inds():
    test_data = []
    most_recent_data = []
    for ind in range(300):
        k = np.random.randint(1,6)
        for _ind in range(k):
            test_data.append({'_id': ind, 'version':_ind})
        most_recent_data.append({'_id': ind, 'version': _ind})
    rng = np.random.RandomState(0)
    p = rng.permutation(len(test_data))
    test_data_rearranged = [test_data[_p] for _p in p]
    rind = get_most_recent_inds(test_data_rearranged)
    test_data_rearranged_most_recent = [test_data_rearranged[idx] for idx in rind]
    assert all([t in most_recent_data for t in test_data_rearranged_most_recent])
    assert len(test_data_rearranged_most_recent) == len(most_recent_data)

    test_data = [{'_id':0, 'version':1}]
    
    assert get_most_recent_inds(test_data).tolist() == [0]
    
    test_data = [{'_id':0, 'version':1}, {'_id':0, 'version':2}]
    assert get_most_recent_inds(test_data).tolist() == [1]
    
    test_data = [{'_id':0, 'version':1}, {'_id':0, 'version':2},
                 {'_id':1, 'version':1}]
    
    assert get_most_recent_inds(test_data).tolist() == [1, 2]
    
    test_data = [{'_id': -1, 'version':1}, {'_id':0, 'version':1},
                 {'_id':0, 'version':2}, {'_id':1, 'version':1}]
    
    assert get_most_recent_inds(test_data).tolist() == [0, 2, 3]
    
    test_data = [{'_id': -1, 'version':1}, {'_id':0, 'version':1},
                 {'_id':0, 'version':2}, {'_id':0, 'version':2}]
    
    assert get_most_recent_inds(test_data).tolist() == [0, 3]