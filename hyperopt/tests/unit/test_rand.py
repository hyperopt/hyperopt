import unittest
from hyperopt.base import Trials, trials_from_docs, miscs_to_idxs_vals
from hyperopt import rand
from hyperopt.tests.test_base import Suggest_API
from .test_domains import gauss_wave2, coin_flip
import numpy as np

TestRand = Suggest_API.make_tst_class(rand.suggest, gauss_wave2(), "TestRand")


class TestRand(unittest.TestCase):
    def test_seeding(self):
        # -- assert that the seeding works a particular way

        domain = coin_flip()
        docs = rand.suggest(
            list(range(10)), domain, Trials(), seed=np.random.PCG64(123)
        )
        trials = trials_from_docs(docs)
        idxs, vals = miscs_to_idxs_vals(trials.miscs)

        # Passes Nov 8 / 2013
        self.assertEqual(list(idxs["flip"]), list(range(10)))
        self.assertEqual(list(vals["flip"]), [0, 1, 1, 0, 1, 0, 0, 0, 0, 0])

    # -- TODO: put in a test that guarantees that
    #          stochastic nodes are sampled in a particular order.
