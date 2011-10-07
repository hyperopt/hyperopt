import logging
import unittest
import sys

import theano
import montetheano
from montetheano.for_theano import where

import hyperopt
import hyperopt.bandits
from hyperopt.bandit_algos import GM_BanditAlgo, TheanoRandom
from hyperopt.experiments import SerialExperiment
import idxs_vals_rnd
from idxs_vals_rnd import IndependentAdaptiveParzenEstimator

def ops(fn, OpCls):
    return [ap for ap in fn.maker.env.toposort()
        if isinstance(ap.op, OpCls)]


def gmms(fn):
    return ops(fn, montetheano.distributions.GMM1)


def bgmms(fn):
    return ops(fn, montetheano.distributions.BGMM1)


def categoricals(fn):
    return ops(fn, montetheano.distributions.Categorical)


def adaptive_parzens(fn):
    return ops(fn, idxs_vals_rnd.AdaptiveParzen)


class TestGM_Distractor(unittest.TestCase): # Tests normal
    def setUp(self):
        self.experiment = SerialExperiment(
            bandit=hyperopt.bandits.Distractor(),
            bandit_algo=GM_BanditAlgo(
                    good_estimator=IndependentAdaptiveParzenEstimator(),
                    bad_estimator=IndependentAdaptiveParzenEstimator()))
        self.experiment.set_bandit()

    def test_op_counts(self):
        # If everything is done right, there should be
        # 2 adaptive parzen estimators in the algorithm
        #  - one for fitting the good examples
        #  - one for fitting the rest of the examples
        # 1 GMM1 Op for drawing from the fit of good examples

        def gmms(fn):
            return [ap for ap in fn.maker.env.toposort()
                if isinstance(ap.op, montetheano.distributions.GMM1)]

        def adaptive_parzens(fn):
            return [ap for ap in fn.maker.env.toposort()
                if isinstance(ap.op, idxs_vals_rnd.AdaptiveParzen)]

        self.experiment.bandit_algo.build_helpers(do_compile=True)
        HL = self.experiment.bandit_algo.helper_locals
        if 1:
            f = theano.function(
                [HL['n_to_draw'], HL['n_to_keep'], HL['y_thresh'], HL['yvals']]
                    + HL['s_obs'].flatten(),
                HL['G_ll'],
                allow_input_downcast=True,
                )
            theano.printing.debugprint(f)
            #theano.printing.pydotprint(f, 'f.png')
            assert len(gmms(f)) == 1
            assert len(adaptive_parzens(f)) == 1

        if 1:
            f = theano.function(
                [HL['n_to_draw'], HL['n_to_keep'], HL['y_thresh'], HL['yvals']]
                    + HL['s_obs'].flatten(),
                HL['G_ll'] - HL['B_ll'],
                allow_input_downcast=True,
                )
            #print gmms(f)
            #print adaptive_parzens(f)
            assert len(gmms(f)) == 1
            assert len(adaptive_parzens(f)) == 2

        self.experiment.bandit_algo.build_helpers(do_compile=True)
        _helper = self.experiment.bandit_algo._helper
        assert len(gmms(_helper)) == 1
        assert len(adaptive_parzens(_helper)) == 2


    def test_optimize_20(self):
        self.experiment.run(50)

        import matplotlib.pyplot as plt
        plt.subplot(1,2,1)
        plt.plot(self.experiment.Ys())
        plt.subplot(1,2,2)
        plt.hist(
                [t['doc'] for t in self.experiment.trials],
                bins=20)
        plt.show()


class TestGM_TwoArms(unittest.TestCase): # Tests one_of
    def setUp(self):
        self.experiment = SerialExperiment(
            bandit=hyperopt.bandits.TwoArms(),
            bandit_algo=GM_BanditAlgo(
                    good_estimator=IndependentAdaptiveParzenEstimator(),
                    bad_estimator=IndependentAdaptiveParzenEstimator()))
        self.experiment.set_bandit()

    def test_optimize_20(self):
        self.experiment.bandit_algo.build_helpers()
        HL = self.experiment.bandit_algo.helper_locals
        assert len(HL['Gsamples']) == 1
        Gpseudocounts = HL['Gsamples'][0].vals.owner.inputs[1]
        Bpseudocounts = HL['Bsamples'][0].vals.owner.inputs[1]

        f = theano.function(
            [HL['n_to_draw'], HL['n_to_keep'], HL['y_thresh'], HL['yvals']]
                + HL['s_obs'].flatten(),
            HL['Gsamples'].take(HL['keep_idxs']).flatten(),
            allow_input_downcast=True,
            )
        debug = theano.function(
            [HL['n_to_draw'], HL['n_to_keep'], HL['y_thresh'], HL['yvals']]
                + HL['s_obs'].flatten(),
            (HL['Gobs'].flatten()
                + [Gpseudocounts]
                + [Bpseudocounts]
                + [HL['yvals'][where(HL['yvals'] < HL['y_thresh'])]]
                + [HL['yvals'][where(HL['yvals'] >= HL['y_thresh'])]]
                ),
            allow_input_downcast=True,
            )
        debug_rval = [None]
        def _helper(*args):
            rval = f(*args)
            debug_rval[0] = debug(*args)
            return rval
        self.experiment.bandit_algo._helper = _helper
        self.experiment.run(200)

        gobs_idxs, gobs_vals, Gpseudo, Bpseudo, Gyvals, Byvals = debug_rval[0]
        print gobs_idxs
        print 'Gpseudo', Gpseudo
        print 'Bpseudo', Bpseudo

        import matplotlib.pyplot as plt
        plt.subplot(1,4,1)
        Xs = [t['doc'] for t in self.experiment.trials]
        Ys = self.experiment.Ys()
        plt.plot(Ys)
        plt.xlabel('time')
        plt.ylabel('loss')

        plt.subplot(1,4,2)
        plt.scatter(Xs,Ys )
        plt.xlabel('X')
        plt.ylabel('loss')

        plt.subplot(1,4,3)
        plt.hist(Xs )
        plt.xlabel('X')
        plt.ylabel('freq')

        plt.subplot(1,4,4)
        plt.hist(Gyvals, bins=20)
        plt.hist(Byvals, bins=20)

        plt.show()


class TestGM_Quadratic1(unittest.TestCase): # Tests uniform
    def setUp(self):
        self.experiment = SerialExperiment(
            bandit=hyperopt.bandits.Quadratic1(),
            bandit_algo=GM_BanditAlgo(
                    good_estimator=IndependentAdaptiveParzenEstimator(),
                    bad_estimator=IndependentAdaptiveParzenEstimator()))
        self.experiment.set_bandit()

    def test_op_counts(self):
        # If everything is done right, there should be
        # 2 adaptive parzen estimators in the algorithm
        #  - one for fitting the good examples
        #  - one for fitting the rest of the examples
        # 1 GMM1 Op for drawing from the fit of good examples

        def gmms(fn):
            return [ap for ap in fn.maker.env.toposort()
                if isinstance(ap.op, montetheano.distributions.BGMM1)]

        def adaptive_parzens(fn):
            return [ap for ap in fn.maker.env.toposort()
                if isinstance(ap.op, idxs_vals_rnd.AdaptiveParzen)]

        self.experiment.bandit_algo.build_helpers(do_compile=True)
        HL = self.experiment.bandit_algo.helper_locals
        if 1:
            f = theano.function(
                [HL['n_to_draw'], HL['n_to_keep'], HL['y_thresh'], HL['yvals']]
                    + HL['s_obs'].flatten(),
                HL['G_ll'],
                allow_input_downcast=True,
                )
            theano.printing.debugprint(f)
            #theano.printing.pydotprint(f, 'f.png')
            assert len(gmms(f)) == 1
            assert len(adaptive_parzens(f)) == 1

        if 1:
            f = theano.function(
                [HL['n_to_draw'], HL['n_to_keep'], HL['y_thresh'], HL['yvals']]
                    + HL['s_obs'].flatten(),
                HL['G_ll'] - HL['B_ll'],
                allow_input_downcast=True,
                )
            #print gmms(f)
            #print adaptive_parzens(f)
            assert len(gmms(f)) == 1
            assert len(adaptive_parzens(f)) == 2

        self.experiment.bandit_algo.build_helpers(do_compile=True)
        _helper = self.experiment.bandit_algo._helper
        assert len(gmms(_helper)) == 1
        assert len(adaptive_parzens(_helper)) == 2


    def test_optimize_20(self):
        self.experiment.run(50)

        import matplotlib.pyplot as plt
        plt.subplot(1,2,1)
        plt.plot(self.experiment.Ys())
        plt.subplot(1,2,2)
        if 0:
            plt.hist(
                    [t['doc'] for t in self.experiment.trials],
                    bins=20)
        else:
            plt.scatter(
                    [t['doc'] for t in self.experiment.trials],
                    range(len(self.experiment.trials)))
        plt.show()


class TestGM_Q1Lognormal(unittest.TestCase): # Tests lognormal
    def setUp(self):
        self.experiment = SerialExperiment(
            bandit=hyperopt.bandits.Q1Lognormal(),
            bandit_algo=GM_BanditAlgo(
                    good_estimator=IndependentAdaptiveParzenEstimator(),
                    bad_estimator=IndependentAdaptiveParzenEstimator()))
        self.experiment.set_bandit()

    def test_optimize_20(self):
        self.experiment.run(50)

        import matplotlib.pyplot as plt
        plt.subplot(1,2,1)
        plt.plot(self.experiment.Ys())
        plt.subplot(1,2,2)
        if 0:
            plt.hist(
                    [t['doc'] for t in self.experiment.trials],
                    bins=20)
        else:
            plt.scatter(
                    [t['doc'] for t in self.experiment.trials],
                    range(len(self.experiment.trials)))
        plt.show()


class TestGM_EggCarton2(unittest.TestCase): # Tests nested search
    def setUp(self):
        self.experiment = SerialExperiment(
            bandit=hyperopt.bandits.EggCarton2(),
            bandit_algo=GM_BanditAlgo(
                    good_estimator=IndependentAdaptiveParzenEstimator(),
                    bad_estimator=IndependentAdaptiveParzenEstimator()))
        self.experiment.set_bandit()

    def test_op_counts_in_llik(self):
        self.experiment.bandit_algo.build_helpers(do_compile=True, mode='FAST_RUN')
        _helper = self.experiment.bandit_algo._helper
        try:
            assert len(gmms(_helper)) == 0
            assert len(bgmms(_helper)) == 2
            assert len(categoricals(_helper)) == 1
            assert len(adaptive_parzens(_helper)) == 2
        except:
            theano.printing.debugprint(_helper)
            raise

    def test_op_counts_in_Gsamples(self):
        self.experiment.bandit_algo.build_helpers(do_compile=False)
        HL = self.experiment.bandit_algo.helper_locals
        f = theano.function(
            [HL['n_to_draw'], HL['n_to_keep'], HL['y_thresh'], HL['yvals']]
                + HL['s_obs'].flatten(),
            HL['Gsamples'].flatten(),
            allow_input_downcast=True,
            mode='FAST_RUN',
            )
        try:
            assert len(bgmms(f)) == 2
            assert len(categoricals(f)) == 1
            assert len(adaptive_parzens(f)) == 2
        except:
            theano.printing.debugprint(f)
            raise

    def test_optimize_20(self):
        self.experiment.run(50)

        import matplotlib.pyplot as plt
        plt.subplot(1,2,1)
        plt.plot(self.experiment.Ys())
        plt.subplot(1,2,2)
        if 0:
            plt.hist(
                    [t['doc'] for t in self.experiment.trials],
                    bins=20)
        else:
            plt.scatter(
                    [t['doc'] for t in self.experiment.trials],
                    range(len(self.experiment.trials)))
        plt.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    class A: pass
    self = A()
    self.experiment = SerialExperiment(
        bandit=hyperopt.bandits.EggCarton2(),
        bandit_algo=GM_BanditAlgo(
                good_estimator=IndependentAdaptiveParzenEstimator(),
                bad_estimator=IndependentAdaptiveParzenEstimator()))
    self.experiment.set_bandit()
    self.experiment.bandit_algo.build_helpers(True)
    HL = self.experiment.bandit_algo.helper_locals
    f = theano.function(
        [HL['n_to_draw'], HL['n_to_keep'], HL['y_thresh'], HL['yvals']]
            + HL['s_obs'].flatten(),
        HL['Gsamples'].take(HL['keep_idxs']).flatten(),
        allow_input_downcast=True,
        )

    debug_outputs = []
    #obs = (HL['Gobs'].flatten() + HL['Bobs'].flatten())
    #samples = (HL['Gsamples'].flatten() + HL['Bsamples'].flatten())

    for Gpost, Bpost in zip(HL['Gsamples'], HL['Bsamples']):
        lpdf = montetheano.rv.lpdf(Bpost.vals, Gpost.vals)
        debug_outputs.extend([Bpost.vals, Gpost.vals, lpdf])

    debug = theano.function(
        [HL['n_to_draw'], HL['n_to_keep'], HL['y_thresh'], HL['yvals']]
            + HL['s_obs'].flatten(),
        debug_outputs,
        allow_input_downcast=True,
        )
    debug_rval = [None]
    def _helper(*args):
        rval = f(*args)
        debug_rval[0] = debug(*args)
        return rval
    self.experiment.bandit_algo._helper = _helper
    for i in xrange(500):
        try:
            self.experiment.run(1)
        except:
            print 'PROBLEM on iter', i
            #theano.printing.debugprint(f)
            raise

    import matplotlib.pyplot as plt
    plt.subplot(1,2,1)
    plt.plot(self.experiment.Ys())
    plt.subplot(1,2,2)
    if 0:
        plt.hist(
                [t['doc'] for t in self.experiment.trials],
                bins=20)
    elif 0:
        plt.scatter(
                [t['doc'] for t in self.experiment.trials],
                range(len(self.experiment.trials)))
    plt.show()
