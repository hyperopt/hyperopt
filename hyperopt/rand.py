"""
Random search - presented as hyperopt.fmin_random
"""
import logging
import numpy as np

import pyll

from .base import miscs_update_idxs_vals

logger = logging.getLogger(__name__)


def suggest(new_ids, domain, trials, seed):
    #logger.debug("in suggest with seed: %s" % (str(seed)))
    #logger.debug('generating trials for new_ids: %s' % str(new_ids))

    rng = np.random.RandomState(seed)
    rval = []
    for ii, new_id in enumerate(new_ids):
        # -- sample new specs, idxs, vals
        idxs, vals = pyll.rec_eval(
            domain.s_idxs_vals,
            memo={
                domain.s_new_ids: [new_id],
                domain.s_rng: rng,
            })
        new_result = domain.new_result()
        new_misc = dict(tid=new_id, cmd=domain.cmd, workdir=domain.workdir)
        miscs_update_idxs_vals([new_misc], idxs, vals)
        rval.extend(trials.new_trial_docs([new_id],
                    [None], [new_result], [new_misc]))
    return rval

# flake8 likes no trailing blank line
