import logging
import unittest
import sys

import numpy
import theano
import montetheano
from montetheano.for_theano import where

import hyperopt
import hyperopt.bandits
from hyperopt.bandit_algos import GM_BanditAlgo, TheanoRandom
from hyperopt.experiments import SerialExperiment
from hyperopt import idxs_vals_rnd
from hyperopt.idxs_vals_rnd import IndependentAdaptiveParzenEstimator
import hyperopt.dbn import Dummy_DBN_Base

def ops(fn, OpCls):
    if isinstance(fn, list):
        return [v.owner for v in montetheano.for_theano.ancestors(fn)
                if v.owner and isinstance(v.owner.op, OpCls)]
    else:
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
            bandit_algo=GM_BanditAlgo(
                    bandit=hyperopt.bandits.Distractor(),
                    good_estimator=IndependentAdaptiveParzenEstimator(),
                    bad_estimator=IndependentAdaptiveParzenEstimator()))

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
            # theano.printing.debugprint(f)
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
        plt.plot(self.experiment.losses())
        plt.subplot(1,2,2)
        plt.hist(
                [t['x'] for t in self.experiment.trials],
                bins=20)

        print self.experiment.losses()
        print 'MIN', min(self.experiment.losses())
        assert min(self.experiment.losses()) < -1.85

        if 0:
            plt.show()


class TestGM_TwoArms(unittest.TestCase): # Tests one_of
    def setUp(self):
        self.experiment = SerialExperiment(
            bandit_algo=GM_BanditAlgo(
                    bandit=hyperopt.bandits.TwoArms(),
                    good_estimator=IndependentAdaptiveParzenEstimator(),
                    bad_estimator=IndependentAdaptiveParzenEstimator()))

    def test_optimize_20(self):
        self.experiment.bandit_algo.build_helpers()
        HL = self.experiment.bandit_algo.helper_locals
        assert len(HL['Gsamples']) == 1
        Gpseudocounts = HL['Gsamples'][0].vals.owner.inputs[1]
        Bpseudocounts = HL['Bsamples'][0].vals.owner.inputs[1]

        f = self.experiment.bandit_algo._helper
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
        Xs = [t['x'] for t in self.experiment.trials]
        Ys = self.experiment.losses()
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

        print self.experiment.losses()
        print 'MIN', min(self.experiment.losses())
        assert min(self.experiment.losses()) < -3.00

        if 0:
            plt.show()


class TestGM_Quadratic1(unittest.TestCase): # Tests uniform
    def setUp(self):
        self.experiment = SerialExperiment(
            bandit_algo=GM_BanditAlgo(
                    bandit=hyperopt.bandits.Quadratic1(),
                    good_estimator=IndependentAdaptiveParzenEstimator(),
                    bad_estimator=IndependentAdaptiveParzenEstimator()))

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
            # theano.printing.debugprint(f)
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
        plt.plot(self.experiment.losses())
        plt.subplot(1,2,2)
        if 0:
            plt.hist(
                    [t['x'] for t in self.experiment.trials],
                    bins=20)
        else:
            plt.scatter(
                    [t['x'] for t in self.experiment.trials],
                    range(len(self.experiment.trials)))
        print self.experiment.losses()
        print 'MIN', min(self.experiment.losses())
        assert min(self.experiment.losses()) < 0.01

        if 0:
            plt.show()


class TestGM_Q1Lognormal(unittest.TestCase): # Tests lognormal
    def setUp(self):
        self.experiment = SerialExperiment(
            bandit_algo=GM_BanditAlgo(
                    bandit=hyperopt.bandits.Q1Lognormal(),
                    good_estimator=IndependentAdaptiveParzenEstimator(),
                    bad_estimator=IndependentAdaptiveParzenEstimator()))

    def test_optimize_20(self):
        self.experiment.run(50)

        import matplotlib.pyplot as plt
        plt.subplot(1,2,1)
        plt.plot(self.experiment.losses())
        plt.subplot(1,2,2)
        if 0:
            plt.hist(
                    [t['x'] for t in self.experiment.trials],
                    bins=20)
        else:
            plt.scatter(
                    [t['x'] for t in self.experiment.trials],
                    range(len(self.experiment.trials)))
        print self.experiment.losses()
        print 'MIN', min(self.experiment.losses())
        assert min(self.experiment.losses()) < .01
        if 0:
            plt.show()


class TestGM_GaussWave2(unittest.TestCase): # Tests nested search
    def setUp(self):
        self.experiment = SerialExperiment(
            bandit_algo=GM_BanditAlgo(
                    bandit=hyperopt.bandits.GaussWave2(),
                    good_estimator=IndependentAdaptiveParzenEstimator(),
                    bad_estimator=IndependentAdaptiveParzenEstimator()))

    def test_op_counts_in_llik(self):
        self.experiment.bandit_algo.build_helpers(do_compile=True, mode='FAST_RUN')
        HL = self.experiment.bandit_algo.helper_locals
        f = theano.function(
                [HL['n_to_draw'], HL['n_to_keep'], HL['y_thresh'], HL['yvals']]
                    + HL['s_obs'].flatten(),
                HL['log_EI'],
                no_default_updates=True,
                mode='FAST_RUN')                 # required for shape inference
        try:
            assert len(gmms(f)) == 0
            assert len(bgmms(f)) == 2            # sampling from good
            assert len(categoricals(f)) == 1     # sampling from good
            assert len(adaptive_parzens(f)) == 4 # fitting both good and bad
        except:
            theano.printing.debugprint(f)
            raise

    def test_op_counts_in_Gsamples(self):
        self.experiment.bandit_algo.build_helpers(do_compile=True, mode='FAST_RUN')
        HL = self.experiment.bandit_algo.helper_locals
        f = theano.function(
                [HL['n_to_draw'], HL['n_to_keep'], HL['y_thresh'], HL['yvals']]
                    + HL['s_obs'].flatten(),
                HL['Gsamples'].flatten(),
                no_default_updates=True,         # allow prune priors
                mode='FAST_RUN')                 # required for shape inference
        try:
            assert len(gmms(f)) == 0
            assert len(bgmms(f)) == 2            # sampling from good
            assert len(categoricals(f)) == 1     # sampling from good
            assert len(adaptive_parzens(f)) == 2 # fitting both good and bad
        except:
            theano.printing.debugprint(f)
            raise


    def test_optimize_20(self):
        self.experiment.run(50)

        import matplotlib.pyplot as plt
        plt.subplot(1,2,1)
        plt.plot(self.experiment.losses())
        plt.subplot(1,2,2)
        plt.scatter(
                [t['x'] for t in self.experiment.trials],
                range(len(self.experiment.trials)))
        print self.experiment.losses()
        print 'MIN', min(self.experiment.losses())
        assert min(self.experiment.losses()) < -1.75
        if 0:
            plt.show()


class TestGM_DummyDBN(unittest.TestCase):
    def setUp(self):
        self.experiment = SerialExperiment(
            bandit_algo=GM_BanditAlgo(
                    bandit=Dummy_DBN_Base(),
                    good_estimator=IndependentAdaptiveParzenEstimator(),
                    bad_estimator=IndependentAdaptiveParzenEstimator()))
        self._old = theano.gof.link.raise_with_op.print_thunk_trace
        theano.gof.link.raise_with_op.print_thunk_trace = True

    def tearDown(self):
        theano.gof.link.raise_with_op.print_thunk_trace = self._old

    def test_optimize_20(self):
        def callback(node, thunk, storage_map, compute_map):
            numeric_outputs = [storage_map[v][0]
                    for v in node.outputs
                    if isinstance(v.type, theano.tensor.TensorType)]
            numeric_inputs = [storage_map[v][0]
                    for v in node.inputs
                    if isinstance(v.type, theano.tensor.TensorType)]

            if not all([numpy.all(numpy.isfinite(n)) for n in numeric_outputs]):
                theano.printing.debugprint(node, depth=8)
                print 'inputs'
                print numeric_inputs
                print 'outputs'
                print numeric_outputs
                raise ValueError('non-finite created in', node)

        mode = theano.Mode(
                optimizer='fast_compile',
                linker=theano.gof.vm.VM_Linker(callback=callback))
        self.experiment.bandit_algo.build_helpers(mode=mode)
        _helper = self.experiment.bandit_algo._helper
        theano.printing.debugprint(_helper)
        for i in range(50):
            print 'ITER', i
            try:
                self.experiment.run(1)
            except:

                raise

        if 0:
            import matplotlib.pyplot as plt
            plt.subplot(1,2,1)
            plt.plot(self.experiment.losses())
            plt.subplot(1,2,2)
            plt.scatter(
                    [t['x'] for t in self.experiment.trials],
                    range(len(self.experiment.trials)))
            plt.show()

