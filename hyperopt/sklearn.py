"""Scikit-learn integration.

This class is based on :class:`sklearn.model_selection._search.BaseSearchCV` and
inspired by :class:sklearn.model_selection._search_successive_halving.BaseSuccessiveHalving`.
"""

import numpy as np
from sklearn.model_selection._search import is_classifier
from sklearn.model_selection._search import BaseSearchCV
from sklearn.utils.multiclass import check_classification_targets, unique_labels
from sklearn.utils.validation import check_array

from hyperopt.base import STATUS_OK, Trials
from hyperopt.fmin import fmin


class HyperoptSearchCV(BaseSearchCV):
    """Hyper-parameter search with hyperopt.

    Parameters
    ----------

    estimator : estimator object
        An object of that type is instantiated for each set of candidate parameters.
        This is assumed to implement the ``scikit-learn`` estimator interface. The
        estimator needs to provide a ``score`` method or ``scoring`` must be passed.

    space : hyperopt.pyll.Apply node or "annotated"
        The set of possible arguments to `fn` is the set of objects
        that could be created with non-zero probability by drawing randomly
        from this stochastic program involving involving hp_<xxx> nodes
        (see `hyperopt.hp` and `hyperopt.pyll_utils`).
        If set to "annotated", will read space using type hint in fn. Ex:
        (`def fn(x: hp.uniform("x", -1, 1)): return x`)

    max_evals : int
        Allow up to this many function evaluations before returning.

    trials : None or base.Trials (or subclass)
        Storage for completed, ongoing, and scheduled evaluation points.  If
        None, then a temporary `base.Trials` instance will be created.  If
        a trials object, then that trials object will be affected by
        side-effect of this call.

    algo : search algorithm
        This object, such as `hyperopt.rand.suggest` and
        `hyperopt.tpe.suggest` provides logic for sequential search of the
        hyperparameter space.

    warm_start : bool, optional (default False)
        When set to True, reuse the solution of the previous ``fit`` call and add
        iterations to the trials object. Otherwise, reset the ``trials``. ``max_evals``
        refers to the total number of iterations in the ``Trials`` object, so use ``set_params``
        to increase the total number.

    scoring : str or callable, optional (default None)
        Strategy to evaluate the performance of the cross-validated model on the test set.

    n_jobs : int, optional (default None)
        Number of jobs to run in parallel. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors.

    refit : bool, optional (default True)
        Refit an estimator using the best found parameters on the whole dataset.

    cv : int, cross-validation generator or an iterable, optional (default None)
        Determines the cross-validation splitting strategy.

    verbose : int, optional (default 0)
        Controls the verbosity.

    pre_dispatch : int or str, optional (default "2*n_jobs")
        Controls the number of jobs that get dispatched during parallel execution. Reducing this
        number can be useful to avoid high memory usage.

    random_state : int, RandomState instance or None, optional (default None)
        Pseudo random number generator state used for random uniform sampling from lists
        instead of ``scipy.stats`` distributions.

    error_score : 'raise' or numeric, optional (default np.nan)
        Value to assign to the score if an error occurs during fitting.

    return_train_score : bool, optional (default False)
        If ``False``, the ``cv_results_`` attribute will not include training scores.

    Attributes
    ----------
    trials_ : Trials
        The trials object.
    """

    _required_parameters = ["estimator", "space", "max_evals"]

    def __init__(
        self,
        estimator,
        space,
        max_evals,
        trials=None,
        algo=None,
        warm_start=False,
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        random_state=None,
        error_score=np.nan,
        return_train_score=False,
    ):
        """Init method."""
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )

        self.space = space
        self.max_evals = max_evals
        self.trials = trials
        self.algo = algo
        self.warm_start = warm_start
        self.random_state = random_state

    def _check_input_parameters(self, X, y=None, groups=None):
        """Run input checks.

        Based on a similar method in :class:`sklearn.model_selection.BaseSuccessiveHalving`.

        Parameters
        ----------

        X : array-like, shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like, shape (n_samples,) or (n_samples, n_output), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set. Only used in conjunction with a "Group" CV
            instance (e.g., :class:`~sklearn.model_selection.GroupKFold`).

        Raises
        ------
        ValueError

            Raised if

            * ``scoring`` is not a string or callable,
            * ``y`` has less than two classes for a classification task,
            * ``y`` contains complex data, or
            * ``refit`` is not boolean.
        """
        if self.scoring is not None and not (
            isinstance(self.scoring, str) or callable(self.scoring)
        ):
            raise ValueError(
                "scoring parameter must be a string, "
                "a callable or None. Multimetric scoring is not "
                "supported."
            )

        check_array(X)
        if is_classifier(self.estimator):
            y = self._validate_data(X="no_validation", y=y)
            check_classification_targets(y)
            labels = unique_labels(y)
            if len(labels) < 2:
                raise ValueError(
                    "Classifier can't train when only one class is present."
                )

        if not isinstance(self.refit, bool):
            raise ValueError(
                f"refit is expected to be a boolean. Got {type(self.refit)} instead."
            )

    def fit(self, X, y=None, groups=None, **fit_params):
        """Run fit with all sets of parameters.

        Parameters
        ----------

        X : array-like, shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like, shape (n_samples,) or (n_samples, n_output), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set. Only used in conjunction with a "Group" CV
            instance (e.g., :class:`~sklearn.model_selection.GroupKFold`).

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of the estimator.

        Returns
        -------

        self : object
            Instance of fitted estimator.
        """
        self._check_input_parameters(
            X=X,
            y=y,
            groups=groups,
        )
        super().fit(X, y=y, groups=groups, **fit_params)

        return self

    def _run_search(self, evaluate_candidates):
        """Run the ``hyperopt`` iterations.

        Parameters
        ----------

        evaluate_candidates : callable
            Callable defined in :class:`sklearn.model_selection._search.BaseSearchCV`
            that trains and scores the model across the cross-validation folds for the
            given parameter space.
        """

        def _evaluate(params):
            results = evaluate_candidates([params])

            return {
                "loss": -results["mean_test_score"][-1],
                "params": params,
                "status": STATUS_OK,
            }

        if not self.warm_start:
            self.trials_ = Trials()
        else:
            if not hasattr(self, "trials_"):
                if self.trials is None:
                    self.trials_ = Trials()
                else:
                    self.trials_ = self.trials

        if isinstance(self.random_state, int):
            seed = np.random.default_rng(self.random_state)
        elif isinstance(self.random_state, np.random.Generator):
            seed = self.random_state
        elif self.random_state is None:
            seed = None
        else:
            raise ValueError(
                "Please supply a `numpy.random.Generator` or integer for `random_state`."
            )

        fmin(
            _evaluate,
            space=self.space,
            algo=self.algo,
            max_evals=self.max_evals,
            rstate=seed,
            trials=self.trials_,
        )
