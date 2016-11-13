from __future__ import print_function
from __future__ import division
from past.utils import old_div
from collections import defaultdict
import unittest
import numpy as np
import numpy.testing as npt
from hyperopt.rdists import (
    loguniform_gen,
    lognorm_gen,
    quniform_gen,
    qloguniform_gen,
    qnormal_gen,
    qlognormal_gen,
)
from scipy import stats
try:
    from scipy.stats.tests.test_continuous_basic import (
        check_cdf_logcdf,
        check_pdf_logpdf,
        check_pdf,
        check_cdf_ppf,
    )
    # from scipy.stats.tests import test_discrete_basic as tdb
except ImportError:

    def check_cdf_logcdf(*args):
        pass

    def check_pdf_logpdf(*args):
        pass

    def check_pdf(*args):
        pass

    def check_cdf_ppf(*args):
        pass


class TestLogUniform(unittest.TestCase):

    def test_cdf_logcdf(self):
        check_cdf_logcdf(loguniform_gen(0, 1), (0, 1), '')
        check_cdf_logcdf(loguniform_gen(0, 1), (-5, 5), '')

    def test_cdf_ppf(self):
        check_cdf_ppf(loguniform_gen(0, 1), (0, 1), '')
        check_cdf_ppf(loguniform_gen(-2, 1), (-5, 5), '')

    def test_pdf_logpdf(self):
        check_pdf_logpdf(loguniform_gen(0, 1), (0, 1), '')
        check_pdf_logpdf(loguniform_gen(low=-4, high=-0.5), (-2, 1), '')

    def test_pdf(self):
        check_pdf(loguniform_gen(0, 1), (0, 1), '')
        check_pdf(loguniform_gen(low=-4, high=-2), (-3, 2), '')

    def test_distribution_rvs(self):
        alpha = 0.01
        loc = 0
        scale = 1
        arg = (loc, scale)
        distfn = loguniform_gen(0, 1)
        D, pval = stats.kstest(distfn.rvs, distfn.cdf, args=arg, N=1000)
        if (pval < alpha):
            npt.assert_(pval > alpha,
                        "D = %f; pval = %f; alpha = %f; args=%s" % (
                            D, pval, alpha, arg))


class TestLogNormal(unittest.TestCase):

    def test_cdf_logcdf(self):
        check_cdf_logcdf(lognorm_gen(0, 1), (0, 1), '')
        check_cdf_logcdf(lognorm_gen(0, 1), (-5, 5), '')

    def test_cdf_ppf(self):
        check_cdf_ppf(lognorm_gen(0, 1), (0, 1), '')
        check_cdf_ppf(lognorm_gen(-2, 1), (-5, 5), '')

    def test_pdf_logpdf(self):
        check_pdf_logpdf(lognorm_gen(0, 1), (0, 1), '')
        check_pdf_logpdf(lognorm_gen(mu=-4, sigma=0.5), (-2, 1), '')

    def test_pdf(self):
        check_pdf(lognorm_gen(0, 1), (0, 1), '')
        check_pdf(lognorm_gen(mu=-4, sigma=2), (-3, 2), '')

    def test_distribution_rvs(self):
        return
        alpha = 0.01
        loc = 0
        scale = 1
        arg = (loc, scale)
        distfn = lognorm_gen(0, 1)
        D, pval = stats.kstest(distfn.rvs, distfn.cdf, args=arg, N=1000)
        if (pval < alpha):
            npt.assert_(pval > alpha,
                        "D = %f; pval = %f; alpha = %f; args=%s" % (
                            D, pval, alpha, arg))


def check_d_samples(dfn, n, rtol=1e-2, atol=1e-2):
    counts = defaultdict(lambda: 0)
    # print 'sample', dfn.rvs(size=n)
    inc = old_div(1.0, n)
    for s in dfn.rvs(size=n):
        counts[s] += inc
    for ii, p in sorted(counts.items()):
        t = np.allclose(dfn.pmf(ii), p, rtol=rtol, atol=atol)
        if not t:
            print(('Error in sampling frequencies', ii))
            print('value\tpmf\tfreq')
            for jj in sorted(counts):
                print(('%.2f\t%.3f\t%.4f' % (
                    jj, dfn.pmf(jj), counts[jj])))
            npt.assert_(t,
                        "n = %i; pmf = %f; p = %f" % (
                            n, dfn.pmf(ii), p))


class TestQUniform(unittest.TestCase):

    def test_smallq(self):
        low, high, q = (0, 1, .1)
        qu = quniform_gen(low, high, q)
        check_d_samples(qu, n=10000)

    def test_bigq(self):
        low, high, q = (-20, -1, 3)
        qu = quniform_gen(low, high, q)
        check_d_samples(qu, n=10000)

    def test_offgrid_int(self):
        qn = quniform_gen(0, 2, 2)
        assert qn.pmf(0) > 0.0
        assert qn.pmf(1) == 0.0
        assert qn.pmf(2) > 0.0
        assert qn.pmf(3) == 0.0
        assert qn.pmf(-1) == 0.0

    def test_offgrid_float(self):
        qn = quniform_gen(0, 1, .2)
        assert qn.pmf(0) > 0.0
        assert qn.pmf(.1) == 0.0
        assert qn.pmf(.2) > 0.0
        assert qn.pmf(.4) > 0.0
        assert qn.pmf(.8) > 0.0
        assert qn.pmf(-.2) == 0.0
        assert qn.pmf(.99) == 0.0
        assert qn.pmf(-.99) == 0.0


class TestQLogUniform(unittest.TestCase):

    def logp(self, x, low, high, q):
        return qloguniform_gen(low, high, q).logpmf(x)

    def test_smallq(self):
        low, high, q = (0, 1, .1)
        qlu = qloguniform_gen(low, high, q)
        check_d_samples(qlu, n=10000)

    def test_bigq(self):
        low, high, q = (-20, 4, 3)
        qlu = qloguniform_gen(low, high, q)
        check_d_samples(qlu, n=10000)

    def test_point(self):
        low, high, q = (np.log(.05), np.log(.15), 0.5)
        qlu = qloguniform_gen(low, high, q)
        check_d_samples(qlu, n=10000)

    def test_2points(self):
        low, high, q = (np.log(.05), np.log(.75), 0.5)
        qlu = qloguniform_gen(low, high, q)
        check_d_samples(qlu, n=10000)

    def test_point_logpmf(self):
        assert np.allclose(self.logp(0, np.log(.25), np.log(.5), 1), 0.0)

    def test_rounding_logpmf(self):
        assert (self.logp(0, np.log(.25), np.log(.75), 1) >
                self.logp(1, np.log(.25), np.log(.75), 1))
        assert (self.logp(-1, np.log(.25), np.log(.75), 1) ==
                self.logp(2, np.log(.25), np.log(.75), 1) ==
                -np.inf)

    def test_smallq_logpmf(self):
        assert (self.logp(0.2, np.log(.16), np.log(.55), .1) >
                self.logp(0.3, np.log(.16), np.log(.55), .1) >
                self.logp(0.4, np.log(.16), np.log(.55), .1) >
                self.logp(0.5, np.log(.16), np.log(.55), .1) >
                -10)

        assert (self.logp(0.1, np.log(.16), np.log(.55), 1) ==
                self.logp(0.6, np.log(.16), np.log(.55), 1) ==
                -np.inf)


class TestQNormal(unittest.TestCase):

    def test_smallq(self):
        mu, sigma, q = (0, 1, .1)
        qn = qnormal_gen(mu, sigma, q)
        check_d_samples(qn, n=10000)

    def test_bigq(self):
        mu, sigma, q = (-20, 4, 3)
        qn = qnormal_gen(mu, sigma, q)
        check_d_samples(qn, n=10000)

    def test_offgrid_int(self):
        qn = qnormal_gen(0, 1, 2)
        assert qn.pmf(0) > 0.0
        assert qn.pmf(1) == 0.0
        assert qn.pmf(2) > 0.0

    def test_offgrid_float(self):
        qn = qnormal_gen(0, 1, .2)
        assert qn.pmf(0) > 0.0
        assert qn.pmf(.1) == 0.0
        assert qn.pmf(.2) > 0.0
        assert qn.pmf(.4) > 0.0
        assert qn.pmf(-.2) > 0.0
        assert qn.pmf(-.4) > 0.0
        assert qn.pmf(.99) == 0.0
        assert qn.pmf(-.99) == 0.0

    def test_numeric(self):
        qn = qnormal_gen(0, 1, 1)
        assert qn.pmf(500) > -np.inf


class TestQLogNormal(unittest.TestCase):

    def test_smallq(self):
        mu, sigma, q = (0, 1, .1)
        qn = qlognormal_gen(mu, sigma, q)
        check_d_samples(qn, n=10000)

    def test_bigq(self):
        mu, sigma, q = (-20, 4, 3)
        qn = qlognormal_gen(mu, sigma, q)
        check_d_samples(qn, n=10000)

    def test_offgrid_int(self):
        mu, sigma, q = (1, 2, 2)
        qn = qlognormal_gen(mu, sigma, q)
        assert qn.pmf(0) > qn.pmf(2) > qn.pmf(20) > 0
        assert qn.pmf(1) == qn.pmf(2 - .001) == qn.pmf(-1) == 0

    def test_offgrid_float(self):
        mu, sigma, q = (-.5, 2, .2)
        qn = qlognormal_gen(mu, sigma, q)
        assert qn.pmf(0) > qn.pmf(.2) > qn.pmf(2) > 0
        assert qn.pmf(.1) == qn.pmf(.2 - .001) == qn.pmf(-.2) == 0

    def test_numeric(self):
        # XXX we don't have a numerically accurate computation for this guy
        # qn = qlognormal_gen(0, 1, 1)
        # assert -np.inf < qn.logpmf(1e-20) < -50
        # assert -np.inf < qn.logpmf(1e20) < -50
        pass
# -- non-empty last line for flake8
