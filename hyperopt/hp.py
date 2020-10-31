"""
Support nicer user syntax:
    from hyperopt import hp
    hp.uniform('x', 0, 1)
"""
from .pyll_utils import hp_choice as choice
from .pyll_utils import hp_randint as randint
from .pyll_utils import hp_pchoice as pchoice

from .pyll_utils import hp_uniform as uniform
from .pyll_utils import hp_uniformint as uniformint
from .pyll_utils import hp_quniform as quniform
from .pyll_utils import hp_loguniform as loguniform
from .pyll_utils import hp_qloguniform as qloguniform

from .pyll_utils import hp_normal as normal
from .pyll_utils import hp_qnormal as qnormal
from .pyll_utils import hp_lognormal as lognormal
from .pyll_utils import hp_qlognormal as qlognormal
