from base import Apply
from base import Literal
from base import as_apply
from base import scope
from base import rec_eval
from base import clone
from base import clone_merge
from base import dfs
from base import toposort
from delayed_eval import Delayed
from .partial import partial

delayed = Delayed(proxy=partial)

# -- adds symbols to scope
import stochastic
