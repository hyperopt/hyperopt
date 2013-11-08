# -- don't import anything here that could import Theano
#    Because theano configures itself on import.... we want to leave
#    room for the Bandit being loaded by mongo-worker to set things up
#    for itself.


from .base import STATUS_STRINGS
from .base import STATUS_NEW
from .base import STATUS_RUNNING
from .base import STATUS_SUSPENDED
from .base import STATUS_OK
from .base import STATUS_FAIL

from .base import JOB_STATES
from .base import JOB_STATE_NEW
from .base import JOB_STATE_RUNNING
from .base import JOB_STATE_DONE
from .base import JOB_STATE_ERROR

from .base import Bandit
from .base import as_bandit
from .base import Ctrl
from .base import Trials
from .base import trials_from_docs
from .base import Domain

from .fmin import fmin
from .fmin import fmin_pass_expr_memo_ctrl
from .fmin import FMinIter
from .fmin import partial
from .fmin import space_eval

# -- syntactic sugar
import hp

# -- exceptions
import exceptions

# -- Import built-in optimization algorithms
import rand
import tpe

__version__ = '0.0.3.dev'

