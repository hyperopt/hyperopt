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
from scipy.stats.tests.test_continuous_basic import (
    check_cdf_logcdf,
    check_pdf_logpdf,
    check_pdf,
    check_cdf_ppf,
    )
from scipy.stats.tests import test_discrete_basic as tdb


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
        D,pval = stats.kstest(distfn.rvs, distfn.cdf, args=arg, N=1000)
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
        alpha = 0.01
        loc = 0
        scale = 1
        arg = (loc, scale)
        distfn = lognorm_gen(0, 1)
        D,pval = stats.kstest(distfn.rvs, distfn.cdf, args=arg, N=1000)
        if (pval < alpha):
            npt.assert_(pval > alpha,
                        "D = %f; pval = %f; alpha = %f; args=%s" % (
                            D, pval, alpha, arg))



def check_d_samples(dfn, n, rtol=1e-2, atol=1e-2):
    counts = defaultdict(lambda: 0)
    #print 'sample', dfn.rvs(size=n)
    inc = 1.0 / n
    for s in dfn.rvs(size=n):
        counts[s] += inc
    for ii, p in sorted(counts.items()):
        t = np.allclose(dfn.pmf(ii), p, rtol=rtol, atol=atol)
        if not t:
            print 'Error in sampling frequencies', ii
            print 'value\tpmf\tfreq'
            for jj in sorted(counts):
                print ('%.2f\t%.3f\t%.4f' % (
                    jj, dfn.pmf(jj), counts[jj]))
            npt.assert_(t,
                "n = %i; pmf = %f; p = %f" % (
                    n, dfn.pmf(ii), p))



class TestQUniform(unittest.TestCase):
    def test_rvs(self):
        for low, high, q in [(0, 1, .1),
                             (-20, -1, 3),]:
            qu = quniform_gen(low, high, q)
            tdb.check_ppf_ppf(qu, ())
            tdb.check_cdf_ppf(qu, (), '')
            try:
                check_d_samples(qu, n=10000)
            except:
                print low, high, q
                raise


class TestQLogUniform(unittest.TestCase):
    def test_rvs(self):
        for low, high, q in [(0, 1, .1),
                             (-20, 4, 3),]:
            qlu = qloguniform_gen(low, high, q)
            tdb.check_ppf_ppf(qlu, ())
            tdb.check_cdf_ppf(qlu, (), '')
            try:
                check_d_samples(qlu, n=10000)
            except:
                print low, high, q
                raise


class TestQNormal(unittest.TestCase):
    def test_rvs(self):
        for mu, sigma, q in [(0, 1, .1),
                             (-20, 3, 3),]:
            qn = qnormal_gen(mu, sigma, q)
            #tdb.check_ppf_ppf(qn, ())
            #tdb.check_cdf_ppf(qn, (), '')
            try:
                check_d_samples(qn, n=10000)
            except:
                print mu, sigma, q
                raise


class TestQLogNormal(unittest.TestCase):
    def test_rvs(self):
        for mu, sigma, q in [(0, 1, .1),
                             (-1, 3, 3),]:
            qn = qlognormal_gen(mu, sigma, q)
            #tdb.check_ppf_ppf(qn, ())
            #tdb.check_cdf_ppf(qn, (), '')
            try:
                check_d_samples(qn, n=10000)
            except:
                print mu, sigma, q
                raise
# -- non-empty last line for flake8
