"""
Random search - presented as hyperopt.fmin_random
"""
import logging
import numpy as np

import pyll

from .base import miscs_update_idxs_vals

logger = logging.getLogger(__name__)


def suggest(new_ids, domain, trials, seed):
    logger.debug("in suggest with seed: %s" % (str(seed)))
    logger.debug('generating trials for new_ids: %s' % str(new_ids))

    # XXX -- Modifying domain by side-effect is bad form :/
    #        domain should be read-only here :/
    #        in fact domain should not have its own seed or rng
    domain_rng = domain.rng
    rng = np.random.RandomState(seed)
    domain.rng = rng
    domain.install_rng_in_s_idxs_vals(rng)
    try:
        rval = []
        for ii, new_id in enumerate(new_ids):
            rng.seed(seed + ii)
            assert domain.rng is rng
            # -- sample new specs, idxs, vals
            idxs, vals = pyll.rec_eval(
                domain.s_idxs_vals,
                memo={domain.s_new_ids: [new_id]})
            new_result = domain.new_result()
            new_misc = dict(tid=new_id, cmd=domain.cmd, workdir=domain.workdir)
            miscs_update_idxs_vals([new_misc], idxs, vals)
            rval.extend(trials.new_trial_docs([new_id],
                        [None], [new_result], [new_misc]))
    finally:
        if domain_rng is not None:
            domain.rng = domain_rng
            domain.install_rng_in_s_idxs_vals(domain_rng)
    return rval

# flake8 likes no trailing blank line
