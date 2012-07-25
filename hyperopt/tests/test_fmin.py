import numpy as np

from hyperopt.pyll_utils import hp_choice
from hyperopt.pyll_utils import hp_uniform
from hyperopt.pyll_utils import hp_loguniform
from hyperopt.pyll_utils import hp_quniform
from hyperopt.pyll_utils import hp_qloguniform
from hyperopt.pyll_utils import hp_normal
from hyperopt.pyll_utils import hp_lognormal
from hyperopt.pyll_utils import hp_qnormal
from hyperopt.pyll_utils import hp_qlognormal

from hyperopt.rand import fmin_random


def test_quadratic1():

    report = fmin_random(
            lambda d: (d['x'] - 3) ** 2,
            hp_uniform('x', -5, 5),
            max_evals=500)

    assert len(report.trials) == 500
    assert abs(report.argmin['x'] - 3.0) < .25

