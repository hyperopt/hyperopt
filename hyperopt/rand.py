"""
Random search - presented as hyperopt.fmin_random
"""

from .base import miscs_update_idxs_vals
import pyll
import hyperopt

def suggest(new_ids, domain, trials, seed=123):

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

