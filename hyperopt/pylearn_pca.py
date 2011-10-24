"""

There is potentially a lot of approaches to PCA, this file may get there eventually.


Elements of this implementation have been borrowed from the MDP toolkit:
    mdp/nodes/pca_nodes.py
"""

#TODO: estimate number of principle components by cross-validation (early stopping)

#TODO: include the original feature means in the `pca` tuple object so that the full transform
# can be saved, applied to new datasets, and approximately inverted.

import numpy
import scipy.linalg

if 0:
    #TODO : put this trick into Theano as an Op
    #       inplace implementation of diag() Op.
    def diag_as_vector(x):
        if x.ndim != 2:
            raise TypeError('this diagonal is implemented only for matrices', x)
        rval = x[0,:min(*x.shape)]
        rval.strides = (rval.strides[0] + x.strides[0],)
        return rval


def pca_from_cov(cov, lower=0, max_components=None, max_energy_fraction=None):
    """Return (eigvals, eigvecs) of data with covariance `cov`.

    The returned eigvals will be a numpy ndarray vector.
    The returned eigvecs will be a numpy ndarray matrix whose *cols* are the eigenvectors.

    This is recommended for retrieving many components from high-dimensional data.

    :param cov: data covariance matrix
    :type cov: a numpy ndarray 

    :returns: (eigvals, eigvecs) of decomposition
    """

    w, v = scipy.linalg.eigh(a=cov, lower=lower)
    # definition of eigh
    #  a * v[:,i] = w[i] * vr[:,i]
    #  v.H * v = identity


    # total variance (vartot) can be computed at this point:
    vartot = w.sum()

    # sort the eigenvals and vecs by decreasing magnitude
    a = numpy.argsort(w)[::-1]
    w = w[a]
    v = v[:,a]

    if max_components != None:
        w = w[:max_components]
        v = v[:, :max_components]

    if max_energy_fraction != None:
        if not (0.0 <= max_energy_fraction <= 1.0):
            raise ValueError('illegal value for max_energy_fraction', max_energy_fraction)
        if max_energy_fraction < 1.0:
            energy = 0
            i = 0
            while (energy < max_energy_fraction * vartot) and (i < len(w)):
                energy += w[i]
                i += 1
            w = w[:(i-1)]
            v = v[:,:(i-1)]
    return w,v


def pca_from_examples(X, max_components=None, max_energy_fraction=None,
        x_centered=False, inplace=False):
    """Return ((eigvals, eigvecs), centered_X) of observations `X` (1-per-row)

    This function exists to wrap several algorithms for getting the principle components.

    :param max_components:
        Return no more than this many principle components.

    :param max_energy_fraction: 
        Return [only] enough components to account for this fraction of the energy (aka
        variance) in the observations.

    :param x_centered:
        True means to consider X as having mean 0 (even if it actually doesn't!)

    :param inplace:
        If False, we copy X before using it. Otherwise we modify it.

    :returns: ((eigvals, eigvecs), centered_X) of PCA decomposition

    """
    if not inplace:
        X = X.copy()
    centered_X = X
    if not x_centered:
        centered_X -= numpy.mean(centered_X, axis=0)
    cov_X = numpy.dot(centered_X.T, centered_X) / (len(X)- 1)
    evals, evecs = pca_from_cov(cov_X, max_components=max_components,
            max_energy_fraction=max_energy_fraction)
    return ((evals, evecs), centered_X)


def pca_whiten((eigvals, eigvecs), centered_X,eps=1e-14):
    """
    Return the projection of X onto it's principle components.  
    
    The return value has the same number of rows as X, but the number of columns is the number
    of principle components.  Columns of the return value have mean 0, variance 1, and are
    uncorrelated.

    :param pca: the (w,v) pair returned by e.g. pca_from_examples(X)

    """
    pca_of_X = numpy.dot(centered_X, eigvecs)
    pca_of_X /= numpy.sqrt(eigvals+eps)
    return pca_of_X

def pca_whiten_inverse((eigvals, eigvecs), whitened_X, eps=1e-14):
    """
    Return an approximate inverse of the `pca_whiten` transform.

    The inverse is not perfect because pca_whitening discards the least-significant components
    of the data.
    """
    return numpy.dot(whitened_X * (numpy.sqrt(eigvals+eps)), eigvecs.T)

def pca_whiten2(pca_from_examples_rval, eps=1e-14):
    """
    Return the projection of X onto it's principle components.  
    
    The return value has the same number of rows as X, but the number of columns is the number
    of principle components.  Columns of the return value have mean 0, variance 1, and are
    uncorrelated.

    .. code-block:: python

        X = data
        (evals, evecs), whitened_X = pca_whiten(
                pca_from_examples(X, max_components=10),
                eps=1e-3)

    :param pca_from_examples_rval: the ((eigvals, eigvecs), centered_X)
            pair returned by e.g. pca_from_examples(X).

    :returns: ((eigvals, eigvecs), whitened_X)

    """
    ((eigvals, eigvecs), centered_X) = pca_from_examples_rval
    pca_of_X = numpy.dot(centered_X, eigvecs)
    pca_of_X /= numpy.sqrt(eigvals+eps)
    return ((eigvals, eigvecs), pca_of_X)

def zca_whiten((eigvals, eigvecs), centered_X):
    """Return the PCA of X but rotated back into the original vector space.

    See also fft_whiten.py
    """
    pca_of_X = pca_whiten((eigvals,eigvecs), centered_X)
    return numpy.dot(pca_of_X, eigvecs.T)


