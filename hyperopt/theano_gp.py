"""
Gaussian-process (GP)-based optimization algorithm using Theano
"""

__authors__   = "James Bergstra"
__copyright__ = "(c) 2011, James Bergstra"
__license__   = "3-clause BSD License"
__contact__   = "github.com/jaberg/hyperopt"

import logging
logger = logging.getLogger(__name__)

import numpy
import scipy.optimize
import theano
from theano import tensor
from theano_linalg import (solve, cholesky, diag, matrix_inverse, det, PSD_hint,
        trace)
import montetheano

from idxs_vals_rnd import IdxsValsList
from theano_bandit_algos import TheanoBanditAlgo

def dots(*args):
    rval = args[0]
    for a in args[1:]:
        rval = theano.tensor.dot(rval, a)
    return rval


def value(x):
    try:
        return x.get_value()
    except AttributeError:
        return x


class SquaredExponentialKernel(object):
    """

    K(x,y) = exp(-0.5 ||x-y||^2 / l^2)

    Attributes:

        log_lenscale - log(2 l^2)

    """

    def __init__(self, l=1, l_min=1e-4, l_max=1000):
        log_l = numpy.log(2 * (l ** 2))
        log_lenscale = theano.shared(log_l)
        if l_min is None:
            log_lenscale_min = None
        else:
            log_lenscale_min = numpy.log(2 * (l_min ** 2))
        if l_max is None:
            log_lenscale_max = None
        else:
            log_lenscale_max = numpy.log(2 * (l_max ** 2))
        self.log_lenscale = log_lenscale
        self.log_lenscale_min = log_lenscale_min
        self.log_lenscale_max = log_lenscale_max
        if self.log_lenscale.ndim != 0:
            raise TypeError('log_lenscale must be scalar', self.log_lenscale)

    def lenscale(self, thing=None):
        if thing is None:
            thing = self.log_lenscale
        return numpy.sqrt(numpy.exp(value(thing)) / 2.0)
    def set_lenscale(self, new_l):
        candidate = numpy.log(2 * (value(new_l) ** 2))
        if candidate < self.log_lenscale_min:
            raise ValueError('lenscale too small')
            candidate = self.log_lenscale_min
        if candidate > self.log_lenscale_max:
            raise ValueError('lenscale too large')
            candidate = self.log_lenscale_max
        self.log_lenscale.set_value(candidate)

    def __str__(self):
        l = self.lenscale()
        (low,high), = self.param_bounds()
        if low is not None:
            low = self.lenscale(low)
        if high is not None:
            high = self.lenscale(high)
        return "%s{l=%s,bounds=(%s,%s)}"%(
                    self.__class__.__name__,
                    str(l), str(low), str(high))

    def params(self):
        return [self.log_lenscale]

    def param_bounds(self):
        return [(self.log_lenscale_min, self.log_lenscale_max)]

    def K(self, x, y):
        ll2 = tensor.exp(self.log_lenscale) #2l^2
        d = ((x**2).sum(axis=1).dimshuffle(0, 'x')
                + (y ** 2).sum(axis=1)
                - 2 * tensor.dot(x, y.T))
        K = tensor.exp(-d / ll2)
        return K


class ExponentialKernel(object):
    """
    K(x,y) = exp(- ||x-y|| / l)

    Attributes:

        log_lenscale - log(l)

    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        if self.log_lenscale.ndim!=0:
            raise TypeError('log_lenscale must be scalar', self.log_lenscale)
    def __str__(self):
        l = numpy.exp(self.log_lenscale.value)
        return "ExponentialKernel{l=%s}"%str(l)

    @classmethod
    def alloc(cls, l=1):
        log_l = numpy.log(l)
        log_lenscale = theano.shared(log_l)
        return cls(log_lenscale=log_lenscale)

    def params(self):
        return [self.log_lenscale]
    def param_bounds(self):
        return [(self.log_lenscale_min, self.log_lenscale_max)]

    def K(self, x, y):
        l = tensor.exp(self.log_lenscale)
        d = ((x**2).sum(axis=1).dimshuffle(0,'x')
                + (y**2).sum(axis=1)
                - 2 * tensor.dot(x, y.T))
        K = tensor.exp(-tensor.sqrt(d)/l)
        return K


class CategoryKernel(SquaredExponentialKernel):
    """
    K(x,y) is 1 if x==y else exp(-1/l)

    The idea is that it's like a SquaredExponentialKernel
    where every point is a distance of 1 from every other one,
    except itself.

    Attributes:

        log_lenscale -

    """
    def K(self, x, y):
        xx = x.reshape((x.shape[0],))
        yy = y.reshape((y.shape[0],))
        xx = xx.dimshuffle(0,'x') # drop cols because there should only be 1
        yy = yy.dimshuffle(0)     # drop cols because there should only be 1

        ll2 = tensor.exp(self.log_lenscale) #2l^2
        d = tensor.neq(xx,yy)
        K = tensor.exp(-d/ll2)
        return K


class GPR_math(object):
    """
    Formulae for Gaussian Process Regression

    # K - the gram matrix of training data
    # y - the vector of training data values
    # var_y - the vector of training data variances

    """
    def __init__(self, x, y, var_y, K_fn, N=None):
        self.x = x
        self.y = tensor.as_tensor_variable(y)
        self.var_y = tensor.as_tensor_variable(var_y)
        self.K_fn = K_fn
        self.K = self.K_fn(x, x)
        if N is None:
            self.N = self.y.shape[0]
        else:
            self.N = N

    def kyn(self):
        """Return tuple: K, y, var_y, N"""
        return self.K, self.y, self.var_y, self.N

    def s_nll(self):
        """ Marginal negative log likelihood of model

        :note: See RW.pdf page 37, Eq. 2.30.
        """
        K, y, var_y, N = self.kyn()
        rK = PSD_hint(K + var_y * tensor.eye(N))
        nll = (0.5 * dots(y, matrix_inverse(rK), y)
                + 0.5 * tensor.log(det(rK))
                + N / 2.0 * tensor.log(2 * numpy.pi))
        return nll

    def s_mean(self, x):
        """Gaussian Process mean at points x"""
        K, y, var_y, N = self.kyn()
        rK = PSD_hint(K + var_y * tensor.eye(N))
        alpha = tensor.dot(matrix_inverse(rK), y)

        K_x = self.K_fn(self.x, x)
        y_x = tensor.dot(alpha, K_x)
        return y_x

    def s_variance(self, x):
        """Gaussian Process variance at points x"""
        K, y, var_y, N = self.kyn()
        rK = PSD_hint(K + var_y * tensor.eye(N))
        L = cholesky(rK)
        K_x = self.K_fn(self.x, x)
        v = solve(L, K_x)
        var_x = 1 - (v**2).sum(axis=0)
        return var_x

    def s_deg_of_freedom(self):
        """
        Degrees of freedom aka "effective number of parameters"
        of kernel smoother.

        Defined pg. 25 of Rasmussen & Williams.
        """
        K, y, var_y, N = self.kyn()
        rK = PSD_hint(K + var_y * tensor.eye(N))
        dof = trace(tensor.dot(K, matrix_inverse(rK)))
        return dof


class GP_BanditAlgo(TheanoBanditAlgo):
    multiplicative_kernels = True
    constant_liar_global_mean = False
    use_cg = False
    params_l2_penalty = 0

    def __init__(self, bandit):
        TheanoBanditAlgo.__init__(self, bandit)
        self.s_prior = IdxsValsList.fromlists(self.s_idxs, self.s_vals)
        self.n_train = tensor.lscalar('n_train')
        self.y_obs = tensor.vector('y_obs')
        self.y_obs_var = tensor.vector('y_obs_var')
        self.x_obs_IVL = self.s_prior.new_like_self()

        self.kernels = []
        self.params = []  # to be populated by self.sparse_gram_matrix()
        self.param_bounds = []  # to be populated by self.sparse_gram_matrix()

        for iv in self.s_prior:
            dist_name = montetheano.rstreams.rv_dist_name(iv.vals)
            if dist_name == 'normal':
                kern = SquaredExponentialKernel()
                self.kernels.append(kern)
                self.params.extend(kern.params())
                self.param_bounds.extend(kern.param_bounds())
            elif dist_name == 'uniform':
                raise NotImplementedError()
            elif dist_name == 'lognormal':
                raise NotImplementedError()
            elif dist_name == 'quantized_lognormal':
                raise NotImplementedError()
            elif dist_name == 'categorical':
                raise NotImplementedError()
            else:
                raise TypeError("unsupported distribution", dist_name)

        self.gprmath = GPR_math(self.x_obs_IVL,
                self.y_obs,
                self.y_obs_var,
                self.K_fn,
                N=self.n_train)
        self.kernels_sealed = True

        self.nll_obs = self.gprmath.s_nll()


        if 0:
            # fit the kernel parameters by gpr.minimize_nll

            # generate half the candidates randomly,
            # and half the candidates from jobs (especially good ones?)
            candidates = s_prior.new_like_self()

            # EI is integral from -inf to thresh of the
            # Gaussian-distributed prediction GP(x)
            thresh = results.min()
            EI = .5 - .5 * tensor.erf((gpr.mean(candidates) - thresh)
                    / tensor.sqrt(gpr.variance(candidates)))

            weighted_EI = EI * tensor.exp(
                    lpdf(candidates)
                    - lpdf(candidates).max())

            g_candidate_vals = grad(weighted_EI.sum(), candidates.valslist())

            # optimize EI.sum() wrt the continuous-valued variables in candidates

    def K_fn(self, x0, x1):
        # for each random variable in self.s_prior
        # choose a kernel to compare observations of that variable
        # and do a sparse increment of the gram matrix
        if self.multiplicative_kernels:
            fill_value = 1
        else:
            fill_value = 0
        gram_matrices = [
                kern.K(
                    #iv0.idxs,
                    iv0.vals.dimshuffle(0, 'x'),
                    #iv1.idxs,
                    iv1.vals.dimshuffle(0, 'x'),
                    #fill_value=fill_value
                    )
                for kern, iv0, iv1 in zip(self.kernels, x0, x1)]

        if self.multiplicative_kernels:
            return tensor.mul(*gram_matrices)
        else:
            return tensor.add(*gram_matrices)

    def prepare_GP_training_data(self, X_IVLs, Ys):
        if self.constant_liar_global_mean:
            y_mean = numpy.mean(Ys['ok'][:self.n_startup_jobs])
        else:
            y_mean = numpy.mean(Ys['ok'])

        x_all = X_IVLs['ok'].copy()
        y_all = list(Ys['ok'])

        for pseudo_bad_status in 'new', 'running':
            logger.info('GM_BanditAlgo assigning bad scores to %i new jobs'
                    % len(Ys[pseudo_bad_status]))
            idmap = x_all.stack(X_IVLs[pseudo_bad_status])
            assert range(len(idmap)) == list(sorted(idmap.keys()))
            y_all.extend([y_thresh + 1 for y in Ys[pseudo_bad_status]])

        # assert that stack() isn't written badly
        assert len(x_all) == len(X_IVLs['ok'])

        y_all = numpy.asarray(y_all)
        return x_all, y_all, y_mean

    def fit_GP(self, x_all, y_all, y_mean, maxiter=None):
        """
        Fit GPR kernel parameters by minimizing magininal nll.

        Returns: None

        Side effect: chooses optimal kernel parameters.
        """

        if list(sorted(x_all.idxset())) != range(len(x_all.idxset())):
            raise NotImplementedError('need contiguous 0-based indexes on x')
        n_train = len(y_all)

        y_all = y_all - y_mean
        self._GP_n_train = n_train
        self._GP_x_all = x_all
        self._GP_y_all = y_all
        self._GP_y_mean = y_mean
        self._GP_y_all_var = numpy.zeros_like(y_all) + 1e-4  #XXX: fit the variance!


        if hasattr(self, 'nll_fn'):
            nll_fn = self.nll_fn
            dnll_dparams = self.dnll_dparams
        else:
            cost = (self.nll_obs
                + self.params_l2_penalty * sum(
                    [(p ** 2).sum() for p in self.params]))
            nll_fn = self.nll_fn = theano.function(
                    [self.n_train, self.y_obs, self.y_obs_var]
                        + self.x_obs_IVL.flatten(),
                    cost,
                    allow_input_downcast=True)
            #theano.printing.debugprint(nll_fn)
            dnll_dparams = self.dnll_dparams = theano.function(
                    [self.n_train, self.y_obs, self.y_obs_var]
                        + self.x_obs_IVL.flatten(),
                    tensor.grad(cost, self.params),
                    allow_input_downcast=True)

        lbounds = []
        ubounds = []
        for lb, ub in self.param_bounds:
            lbounds.extend(numpy.asarray(value(lb)).flatten())
            ubounds.extend(numpy.asarray(value(ub)).flatten())
        bounds = numpy.asarray([lbounds, ubounds]).T

        def get_pt():
            #XXX: handle non-scalar parameters...
            rval = []
            for p in self.params:
                v = p.get_value().flatten()
                rval.extend(v)
            return numpy.asarray(rval)
        def set_pt(pt):
            i = 0
            for p in self.params:
                shape = p.get_value(borrow=True).shape
                size = numpy.prod(shape)
                p.set_value(pt[i:i + size].reshape(shape))
                i += size
            assert i == len(pt)
            #print self.kernel.summary()

        def f(pt):
            #print 'f', pt
            set_pt(pt)
            # XXX TODO: estimate y_obs_var
            return nll_fn(self._GP_n_train,
                    self._GP_y_all,
                    self._GP_y_all_var,
                    *self._GP_x_all.flatten())
        def df(pt):
            #print 'df', pt
            set_pt(pt)
            dparams = dnll_dparams(self._GP_n_train,
                    self._GP_y_all,
                    self._GP_y_all_var,
                    *self._GP_x_all.flatten())
            rval = []
            for dp in dparams:
                rval.extend(dp.flatten())
            rval =  numpy.asarray(rval)
            #print numpy.sqrt((rval**2).sum())
            return rval

        # XXX: re-initialize current point to a reasonable randomized default
        #      Remi found warm-starting was a bad idea

        start_pt = get_pt()
        if self.use_cg:
            # WEIRD: I was using fmin_ncg here until I used a low multiplier on
            # the sum-squared-error regularizer on the 'cost' above, which threw
            # ncg into an inf loop!?
            best_pt = scipy.optimize.fmin_cg(f, start_pt, df,
                    maxiter=maxiter,
                    epsilon=.02)
        else:
            best_pt, best_value, best_d = scipy.optimize.fmin_l_bfgs_b(f,
                    start_pt,
                    df,
                    maxfun=maxiter,
                    bounds=bounds,
                    iprint=1)
        logger.info('fit_GP best value: %f' % best_value)
        set_pt(best_pt)
        return best_value

    def GP_mean(self, x):
        """
        Compute mean at points in x_new
        """
        return self.GP_mean_variance(x)[0]

    def GP_variance(self, x):
        """
        Compute variance at points in x_new
        """
        return self.GP_mean_variance(x)[1]

    def GP_mean_variance(self, x):
        """
        Compute mean and variance at points in x_new
        """
        try:
            self._mean_variance
        except AttributeError:
            s_x = self.s_prior.new_like_self()
            self._mean_variance = theano.function(
                    [self.n_train, self.y_obs, self.y_obs_var]
                        + self.x_obs_IVL.flatten()
                        + s_x.flatten(),
                    [self.gprmath.s_mean(s_x), self.gprmath.s_variance(s_x)],
                    allow_input_downcast=True)
        if len(x) != len(self._GP_x_all):
            raise ValueError('x has wrong len')
        rval_mean, rval_var = self._mean_variance(
                self._GP_n_train,
                self._GP_y_all,
                self._GP_y_all_var,
                *(self._GP_x_all.flatten() + x.flatten()))
        return rval_mean + self._GP_y_mean, rval_var

    def theano_suggest_from_model(self, X_IVLs, Ys, N):
        raise NotImplementedError()

    mode = None  # to use theano's default

    n_startup_jobs = 30  # enough to estimate mean and variance in Y | prior(X)
                         # should be bandit-agnostic

    def theano_suggest_from_prior(self, N):
        if not hasattr(self, '_prior_sampler'):
            self._prior_sampler = theano.function(
                    [self.s_N],
                    self.s_prior.flatten(),
                    mode=self.mode)
        rvals = self._prior_sampler(N)
        return IdxsValsList.fromflattened(rvals)

    def theano_suggest(self, X_IVLs, Ys, N):
        if len(Ys['ok']) < self.n_startup_jobs:
            return self.theano_suggest_from_prior(N)
        else:
            return self.theano_suggest_from_model(X_IVLs, Ys, N)



class GPR_algos(object):

    def deg_of_freedom(self):
        try:
            self._dof_fn
        except AttributeError:
            self._dof_fn = theano.function([], self.s_deg_of_freedom())
        return self._dof_fn()



