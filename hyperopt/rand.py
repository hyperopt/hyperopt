"""
Random search - presented as hyperopt.fmin_random
"""
import logging

import pyll
import hyperopt

from .base import miscs_update_idxs_vals

logger = logging.getLogger(__name__)


def suggest(new_ids, domain, trials, seed=123):
    logger.info('generating trials for new_ids: %s' % str(new_ids))

    rval = []
    for new_id in new_ids:
        # -- hack - domain should be read-only here :/
        #    in fact domain should not have its own seed or rng
        domain.rng.seed(seed + new_id)
        # -- sample new specs, idxs, vals
        idxs, vals = pyll.rec_eval(domain.s_idxs_vals,
                memo={domain.s_new_ids: [new_id]})
        new_result = domain.new_result()
        new_misc = dict(tid=new_id, cmd=domain.cmd, workdir=domain.workdir)
        miscs_update_idxs_vals([new_misc], idxs, vals)
        rval.extend(trials.new_trial_docs([new_id],
                [None], [new_result], [new_misc]))
    return rval

