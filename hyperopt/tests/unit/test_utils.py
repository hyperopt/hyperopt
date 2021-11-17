import numpy as np
from nose.tools import raises, eq_
import shutil
import os
from hyperopt.utils import fast_isin
from hyperopt.utils import get_most_recent_inds
from hyperopt.utils import temp_dir, working_dir, get_closest_dir, path_split_all


def test_fast_isin():
    Y = np.random.randint(0, 10000, size=(100,))
    X = np.arange(10000)
    Z = fast_isin(X, Y)
    D = np.unique(Y)
    D.sort()
    T1 = (X[Z] == D).all()

    X = np.array(list(range(10000)) + list(range(10000)))
    Z = fast_isin(X, Y)
    T2 = (X[Z] == np.append(D, D.copy())).all()

    X = np.random.randint(0, 100, size=(40,))
    X.sort()
    Y = np.random.randint(0, 100, size=(60,))
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
        k = np.random.randint(1, 6)
        for _ind in range(k):
            test_data.append({"_id": ind, "version": _ind})
        most_recent_data.append({"_id": ind, "version": _ind})
    rng = np.random.default_rng(0)
    p = rng.permutation(len(test_data))
    test_data_rearranged = [test_data[_p] for _p in p]
    rind = get_most_recent_inds(test_data_rearranged)
    test_data_rearranged_most_recent = [test_data_rearranged[idx] for idx in rind]
    assert all([t in most_recent_data for t in test_data_rearranged_most_recent])
    assert len(test_data_rearranged_most_recent) == len(most_recent_data)

    test_data = [{"_id": 0, "version": 1}]

    assert get_most_recent_inds(test_data).tolist() == [0]

    test_data = [{"_id": 0, "version": 1}, {"_id": 0, "version": 2}]
    assert get_most_recent_inds(test_data).tolist() == [1]

    test_data = [
        {"_id": 0, "version": 1},
        {"_id": 0, "version": 2},
        {"_id": 1, "version": 1},
    ]

    assert get_most_recent_inds(test_data).tolist() == [1, 2]

    test_data = [
        {"_id": -1, "version": 1},
        {"_id": 0, "version": 1},
        {"_id": 0, "version": 2},
        {"_id": 1, "version": 1},
    ]

    assert get_most_recent_inds(test_data).tolist() == [0, 2, 3]

    test_data = [
        {"_id": -1, "version": 1},
        {"_id": 0, "version": 1},
        {"_id": 0, "version": 2},
        {"_id": 0, "version": 2},
    ]

    assert get_most_recent_inds(test_data).tolist() == [0, 3]


@raises(RuntimeError)
def test_temp_dir_pardir():
    with temp_dir("../test_temp_dir"):
        pass


def test_temp_dir():
    fn = "test_temp_dir"
    if os.path.exists(fn):
        print("Path %s exists, not running test_temp_dir()" % fn)
        return
    try:
        assert not os.path.exists(fn)
        with temp_dir(fn):
            assert os.path.exists(fn)
        assert os.path.exists(fn)
        os.rmdir(fn)

        assert not os.path.exists(fn)
        with temp_dir(fn, erase_after=True):
            assert os.path.exists(fn)
        assert not os.path.exists(fn)
    finally:
        if os.path.isdir(fn):
            os.rmdir(fn)


def test_path_split_all():
    ll = "foo bar baz".split()
    path = os.path.join(*ll)
    eq_(list(path_split_all(path)), ll)


def test_temp_dir_sentinel():
    from os.path import join, isdir, exists

    basedir = "test_temp_dir_sentinel"
    fn = join(basedir, "foo", "bar")
    if exists(basedir):
        print("Path %s exists, not running test_temp_dir_sentinel()" % basedir)
        return
    os.makedirs(basedir)
    eq_(get_closest_dir(fn)[0], basedir)
    eq_(get_closest_dir(fn)[1], "foo")
    sentinel = join(basedir, "foo.inuse")
    try:
        with temp_dir(fn, erase_after=True, with_sentinel=True):
            assert isdir(fn)
            assert exists(sentinel)
            # simulate work
            open(join(fn, "dummy.txt"), "w").close()
        # work file should be deleted together with directory
        assert not exists(fn)
        assert not exists(join(basedir, "foo"))
        # basedir should still exist, though!
        assert isdir(basedir)
    finally:
        if isdir(basedir):
            shutil.rmtree(basedir)


def test_workdir():
    fn = "test_work_dir"
    os.makedirs(fn)
    try:
        assert fn not in os.getcwd()
        with working_dir(fn):
            assert fn in os.getcwd()
        assert fn not in os.getcwd()
    finally:
        if os.path.isdir(fn):
            os.rmdir(fn)
