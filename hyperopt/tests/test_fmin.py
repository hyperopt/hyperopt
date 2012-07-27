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

from hyperopt import fmin, algo


def test_quadratic1():

    report = fmin(
            fn=lambda x: (x - 3) ** 2,
            space=hp_uniform('x', -5, 5),
            algo=algo.random,
            max_evals=500)

    assert len(report.trials) == 500
    assert abs(report.trials.argmin['x'] - 3.0) < .25

