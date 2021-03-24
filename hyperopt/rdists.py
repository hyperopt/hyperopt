"""
Extra distributions to complement scipy.stats

"""
from past.utils import old_div
import numpy as np
import numpy.random as mtrand
import scipy.stats
from scipy.stats import rv_continuous  # , rv_discrete
from scipy.stats._continuous_distns import lognorm_gen as scipy_lognorm_gen


class loguniform_gen(rv_continuous):
    """Stats for Y = e^X where X ~ U(low, high)."""

    def __init__(self, low=0, high=1):
        rv_continuous.__init__(self, a=np.exp(low), b=np.exp(high))
        self._low = low
        self._high = high

    def _rvs(self):
        rval = np.exp(mtrand.uniform(self._low, self._high, self._size))
        return rval

    def _pdf(self, x):
        return old_div(1.0, (x * (self._high - self._low)))

    def _logpdf(self, x):
        return -np.log(x) - np.log(self._high - self._low)

    def _cdf(self, x):
        return old_div((np.log(x) - self._low), (self._high - self._low))


class lognorm_gen(scipy_lognorm_gen):
    def __init__(self, mu, sigma):
        self.mu_ = mu
        self.s_ = sigma
        scipy_lognorm_gen.__init__(self)

        # I still don't understand what scipy stats objects are doing
        # re: this stuff
        del self.__dict__["_parse_args"]
        del self.__dict__["_parse_args_stats"]
        del self.__dict__["_parse_args_rvs"]

    def _parse_args(self, *args, **kwargs):
        assert not args, args
        assert not kwargs, kwargs
        args = (self.s_,)
        loc = 0
        scale = np.exp(self.mu_)
        return args, loc, scale


def qtable_pmf(x, q, qlow, xs, ps):
    qx = np.round(old_div(np.atleast_1d(x).astype(float), q)) * q
    is_multiple = np.isclose(qx, x)
    ix = np.round(old_div((qx - qlow), q)).astype(int)
    is_inbounds = np.logical_and(ix >= 0, ix < len(ps))
    oks = np.logical_and(is_multiple, is_inbounds)
    rval = np.zeros_like(qx)
    rval[oks] = np.asarray(ps)[ix[oks]]
    if isinstance(x, np.ndarray):
        return rval.reshape(x.shape)
    return float(rval)


def qtable_logpmf(x, q, qlow, xs, ps):
    p = qtable_pmf(np.atleast_1d(x), q, qlow, xs, ps)
    # -- this if/else avoids np warning about underflow
    rval = np.zeros_like(p)
    rval[p == 0] = -np.inf
    rval[p != 0] = np.log(p[p != 0])
    if isinstance(x, np.ndarray):
        return rval
    return float(rval)


class quniform_gen:
    # -- not inheriting from scipy.stats.rv_discrete
    #    because I don't understand the design of those rv classes
    """Stats for Y = q * round(X / q) where X ~ U(low, high)."""

    def __init__(self, low, high, q):
        low, high = list(map(float, (low, high)))
        qlow = safe_int_cast(np.round(old_div(low, q))) * q
        qhigh = safe_int_cast(np.round(old_div(high, q))) * q
        if qlow == qhigh:
            xs = [qlow]
            ps = [1.0]
        else:
            lowmass = 1 - (old_div((low - qlow + 0.5 * q), q))
            assert 0 <= lowmass <= 1.0, (lowmass, low, qlow, q)
            highmass = old_div((high - qhigh + 0.5 * q), q)
            assert 0 <= highmass <= 1.0, (highmass, high, qhigh, q)
            # -- xs: qlow to qhigh inclusive
            xs = np.arange(qlow, qhigh + 0.5 * q, q)
            ps = np.ones(len(xs))
            ps[0] = lowmass
            ps[-1] = highmass
            ps /= ps.sum()

        self.low = low
        self.high = high
        self.q = q
        self.qlow = qlow
        self.qhigh = qhigh
        self.xs = np.asarray(xs)
        self.ps = np.asarray(ps)

    def pmf(self, x):
        return qtable_pmf(x, self.q, self.qlow, self.xs, self.ps)

    def logpmf(self, x):
        return qtable_logpmf(x, self.q, self.qlow, self.xs, self.ps)

    def rvs(self, size=()):
        rval = mtrand.uniform(low=self.low, high=self.high, size=size)
        rval = safe_int_cast(np.round(old_div(rval, self.q))) * self.q
        return rval


class qloguniform_gen(quniform_gen):
    """Stats for Y = q * round(e^X / q) where X ~ U(low, high)."""

    # -- not inheriting from scipy.stats.rv_discrete
    #    because I don't understand the design of those rv classes

    def __init__(self, low, high, q):
        low, high = list(map(float, (low, high)))
        elow = np.exp(low)
        ehigh = np.exp(high)
        qlow = safe_int_cast(np.round(old_div(elow, q))) * q
        qhigh = safe_int_cast(np.round(old_div(ehigh, q))) * q

        # -- loguniform for using the CDF
        lu = loguniform_gen(low=low, high=high)

        cut_low = np.exp(low)  # -- lowest possible pre-round value
        cut_high = min(
            qlow + 0.5 * q, ehigh  # -- highest value that would ...
        )  # -- round to qlow
        xs = [qlow]
        ps = [lu.cdf(cut_high)]
        ii = 0
        cdf_high = ps[0]

        while cut_high < (ehigh - 1e-10):
            # TODO: cut_low never used
            cut_high, cut_low = min(cut_high + q, ehigh), cut_high
            cdf_high, cdf_low = lu.cdf(cut_high), cdf_high
            ii += 1
            xs.append(qlow + ii * q)
            ps.append(cdf_high - cdf_low)

        ps = np.asarray(ps)
        ps /= ps.sum()

        self.low = low
        self.high = high
        self.q = q
        self.qlow = qlow
        self.qhigh = qhigh
        self.xs = np.asarray(xs)
        self.ps = ps

    def pmf(self, x):
        return qtable_pmf(x, self.q, self.qlow, self.xs, self.ps)

    def logpmf(self, x):
        return qtable_logpmf(x, self.q, self.qlow, self.xs, self.ps)

    def rvs(self, size=()):
        x = mtrand.uniform(low=self.low, high=self.high, size=size)
        rval = safe_int_cast(np.round(old_div(np.exp(x), self.q))) * self.q
        return rval


class qnormal_gen:
    """Stats for Y = q * round(X / q) where X ~ N(mu, sigma)"""

    def __init__(self, mu, sigma, q):
        self.mu, self.sigma = list(map(float, (mu, sigma)))
        self.q = q
        # -- distfn for using the CDF
        self._norm_logcdf = scipy.stats.norm(loc=mu, scale=sigma).logcdf

    def in_domain(self, x):
        return np.isclose(x, safe_int_cast(np.round(old_div(x, self.q))) * self.q)

    def pmf(self, x):
        return np.exp(self.logpmf(x))

    def logpmf(self, x):
        x1 = np.atleast_1d(x)
        in_domain = self.in_domain(x1)
        rval = np.zeros_like(x1, dtype=float) - np.inf
        x_in_domain = x1[in_domain]

        ubound = x_in_domain + self.q * 0.5
        lbound = x_in_domain - self.q * 0.5
        # -- reflect intervals right of mu to other side
        #    for more accurate calculation
        flip = lbound > self.mu
        tmp = lbound[flip].copy()
        lbound[flip] = self.mu - (ubound[flip] - self.mu)
        ubound[flip] = self.mu - (tmp - self.mu)

        assert np.all(ubound > lbound)
        a = self._norm_logcdf(ubound)
        b = self._norm_logcdf(lbound)
        rval[in_domain] = a + np.log1p(-np.exp(b - a))
        if isinstance(x, np.ndarray):
            return rval
        return float(rval)

    def rvs(self, size=()):
        x = mtrand.normal(loc=self.mu, scale=self.sigma, size=size)
        rval = safe_int_cast(np.round(old_div(x, self.q))) * self.q
        return rval


class qlognormal_gen:
    """Stats for Y = q * round(exp(X) / q) where X ~ N(mu, sigma)"""

    def __init__(self, mu, sigma, q):
        self.mu, self.sigma = list(map(float, (mu, sigma)))
        self.q = q
        # -- distfn for using the CDF
        self._norm_cdf = scipy.stats.norm(loc=mu, scale=sigma).cdf

    def in_domain(self, x):
        return np.logical_and(
            (x >= 0),
            np.isclose(x, safe_int_cast(np.round(old_div(x, self.q))) * self.q),
        )

    def pmf(self, x):
        x1 = np.atleast_1d(x)
        in_domain = self.in_domain(x1)
        x1_in_domain = x1[in_domain]
        rval = np.zeros_like(x1, dtype=float)
        rval_in_domain = self._norm_cdf(np.log(x1_in_domain + 0.5 * self.q))
        rval_in_domain[x1_in_domain != 0] -= self._norm_cdf(
            np.log(x1_in_domain[x1_in_domain != 0] - 0.5 * self.q)
        )
        rval[in_domain] = rval_in_domain
        if isinstance(x, np.ndarray):
            return rval
        return float(rval)

    def logpmf(self, x):
        pmf = self.pmf(np.atleast_1d(x))
        assert np.all(pmf >= 0)
        pmf[pmf == 0] = -np.inf
        pmf[pmf > 0] = np.log(pmf[pmf > 0])
        if isinstance(x, np.ndarray):
            return pmf
        return float(pmf)

    def rvs(self, size=()):
        x = mtrand.normal(loc=self.mu, scale=self.sigma, size=size)
        rval = safe_int_cast(np.round(old_div(np.exp(x), self.q))) * self.q
        return rval


def safe_int_cast(obj):
    if isinstance(obj, np.ndarray):
        return obj.astype("int")
    if isinstance(obj, list):
        return [int(i) for i in obj]
    return int(obj)


# -- non-empty last line for flake8
