"""
Extra distributions to complement scipy.stats

"""
import numpy as np
import numpy.random as mtrand
import scipy.stats
from scipy.stats import rv_continuous, rv_discrete
from scipy.stats.distributions import rv_generic


class uniform_gen(scipy.stats.distributions.uniform_gen):
    # -- included for completeness
    pass


class norm_gen(scipy.stats.distributions.norm_gen):
    # -- included for completeness
    pass


class loguniform_gen(rv_continuous):
    """ Stats for Y = e^X where X ~ U(low, high).

    """
    def __init__(self, low=0, high=1):
        rv_continuous.__init__(self,
                a=np.exp(low),
                b=np.exp(high))
        self._low = low
        self._high = high

    def _rvs(self):
        rval = np.exp(mtrand.uniform(
            self._low,
            self._high,
            self._size))
        return rval

    def _pdf(self, x):
        return 1.0 / (x * (self._high - self._low))

    def _logpdf(self, x):
        return - np.log(x) - np.log(self._high - self._low)

    def _cdf(self, x):
        return (np.log(x) - self._low) / (self._high - self._low)


# -- cut and paste from scipy.stats
#    because the way s is passed to these functions makes it impossible
#    to construct this class. insane
class lognorm_gen(rv_continuous):
    """A lognormal continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `lognorm` is::

        lognorm.pdf(x, s) = 1 / (s*x*sqrt(2*pi)) * exp(-1/2*(log(x)/s)**2)

    for ``x > 0``, ``s > 0``.

    If log x is normally distributed with mean mu and variance sigma**2,
    then x is log-normally distributed with shape paramter sigma and scale
    parameter exp(mu).

    %(example)s

    """
    def __init__(self, mu, sigma):
        self.mu_ = mu
        self.s_ = sigma
        self.norm_ = scipy.stats.norm
        rv_continuous.__init__(self, a=0.0, name='loguniform', shapes='s')

    def _rvs(self):
        s = self.s_
        return np.exp(self.mu_ + s * self.norm_.rvs(size=self._size))

    def _pdf(self, x):
        s = self.s_
        Px = np.exp(-(np.log(x) - self.mu_ ) ** 2 / (2 * s ** 2))
        return Px / (s * x * np.sqrt(2 * np.pi))

    def _cdf(self, x):
        s = self.s_
        return self.norm_.cdf((np.log(x) - self.mu_) / s)

    def _ppf(self, q):
        s = self.s_
        return np.exp(s*self.norm_._ppf(q) + self.mu_)

    def _stats(self):
        if self.mu_ != 0.0:
            raise NotImplementedError()
        s = self.s_
        p = np.exp(s*s)
        mu = np.sqrt(p)
        mu2 = p*(p-1)
        g1 = np.sqrt((p-1))*(2+p)
        g2 = np.polyval([1,2,3,0,-6.0],p)
        return mu, mu2, g1, g2

    def _entropy(self):
        if self.mu_ != 0.0:
            raise NotImplementedError()
        s = self.s_
        return 0.5 * (1 + np.log(2 * np.pi) + 2 * np.log(s))


def qtable_pmf(x, q, qlow, xs, ps):
    qx = np.round(x / q) * q
    if not np.isclose(qx, x):
        return 0.0
    ix = int(np.round((qx - qlow) / q))
    if 0 <= ix < len(ps):
        assert np.isclose(x, xs[ix])
        return ps[ix]
    else:
        return 0.0


def qtable_logpmf(x, q, qlow, xs, ps):
    p = qtable_pmf(x, q, qlow, xs, ps)
    # -- this if/else avoids np warning about underflow
    if p == 0:
        return -np.inf
    else:
        return np.log(p)


class quniform_gen(object):
    # -- not inheriting from scipy.stats.rv_discrete
    #    because I don't understand the design of those rv classes
    """ Stats for Y = q * round(X / q) where X ~ U(low, high).

    """
    def __init__(self, low, high, q):
        low, high, q = map(float, (low, high, q))
        qlow = np.round(low / q) * q
        qhigh = np.round(high / q) * q
        if qlow == qhigh:
            xs = [qlow]
            ps = [1.0]
        else:
            lowmass = 1 - ((low - qlow + .5 * q) / q)
            assert 0 <= lowmass <= 1.0, (lowmass, low, qlow, q)
            highmass = (high - qhigh + .5 * q) / q
            assert 0 <= highmass <= 1.0, (highmass, high, qhigh, q)
            # -- xs: qlow to qhigh inclusive
            xs = np.arange(qlow, qhigh + .5 * q, q)
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
        rval = np.round(rval / self.q) * self.q
        return rval


class qloguniform_gen(quniform_gen):
    """ Stats for Y = q * round(e^X / q) where X ~ U(low, high).

    """
    # -- not inheriting from scipy.stats.rv_discrete
    #    because I don't understand the design of those rv classes

    def __init__(self, low, high, q):
        low, high, q = map(float, (low, high, q))
        elow = np.exp(low)
        ehigh = np.exp(high)
        qlow = np.round(elow / q) * q
        qhigh = np.round(ehigh / q) * q

        # -- loguniform for using the CDF
        lu = loguniform_gen(low=low, high=high)

        cut_low = np.exp(low) # -- lowest possible pre-round value
        cut_high = min(qlow + .5 * q, # -- highest value that would ...
                       ehigh)         # -- round to qlow
        xs = [qlow]
        ps = [lu.cdf(cut_high)]
        ii = 0
        cdf_high = ps[0]

        while cut_high < (ehigh - 1e-10):
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
        rval = np.round(np.exp(x) / self.q) * self.q
        return rval


class qnormal_gen(object):
    """Stats for Y = q * round(X / q) where X ~ N(mu, sigma)
    """
    def __init__(self, mu, sigma, q):
        self.mu, self.sigma, self.q = map(float, (mu, sigma, q))
        # -- distfn for using the CDF
        self._norm_logcdf = scipy.stats.norm(loc=mu, scale=sigma).logcdf

    def in_domain(self, x):
        return np.allclose(x, np.round(x / self.q) * self.q)

    def pmf(self, x):
        return np.exp(self.logpmf(x))

    def logpmf(self, x):
        if not self.in_domain(x):
            return -np.inf

        ubound = x + self.q * 0.5
        lbound = x - self.q * 0.5
        # -- reflect intervals right of mu to other side
        #    for more accurate calculation
        if lbound > self.mu:
            lbound, ubound = (self.mu - (ubound - self.mu),
                              self.mu - (lbound - self.mu))
        assert ubound > lbound
        a = self._norm_logcdf(ubound)
        b = self._norm_logcdf(lbound)
        return a + np.log1p(- np.exp(b - a))

    def rvs(self, size=()):
        x = mtrand.normal(loc=self.mu, scale=self.sigma, size=size)
        rval = np.round(x / self.q) * self.q
        return rval


class qlognormal_gen(object):
    """Stats for Y = q * round(exp(X) / q) where X ~ N(mu, sigma)
    """
    def __init__(self, mu, sigma, q):
        self.mu, self.sigma, self.q = map(float, (mu, sigma, q))
        # -- distfn for using the CDF
        self._norm_cdf = scipy.stats.norm(loc=mu, scale=sigma).cdf

    def in_domain(self, x):
        return ((x >= 0) & np.allclose(x, np.round(x / self.q) * self.q))

    def pmf(self, x):
        if self.in_domain(x):
            a = self._norm_cdf(np.log(x + 0.5 * self.q))
            if x == 0:
                return a
            else:
                b = self._norm_cdf(np.log(x - 0.5 * self.q))
                return a - b
        else:
            return 0.0

    def logpmf(self, x):
        if self.in_domain(x):
            return np.log(self.pmf(x))
        else:
            return -np.inf

    def rvs(self, size=()):
        x = mtrand.normal(loc=self.mu, scale=self.sigma, size=size)
        rval = np.round(np.exp(x) / self.q) * self.q
        return rval


# -- non-empty last line for flake8
