"""Test scikit-learn integration."""

import pytest
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.utils.estimator_checks import check_estimator

from hyperopt import hp
from hyperopt.sklearn import HyperoptSearchCV


@pytest.mark.parametrize(
    "estimator,check",
    list(
        check_estimator(
            HyperoptSearchCV(
                estimator=Ridge(),
                space={"alpha": hp.uniform("alpha", 0, 1)},
                max_evals=10,
                random_state=42,
            ),
            generate_only=True,
        )
    ),
)
def test_estimator_regression(estimator, check):
    """Test compatibility with the scikit-learn API for regressors."""
    if "predict" in check.func.__name__:
        # Predict methods do a simple passthrough to the underlying best estimator
        # https://github.com/scikit-learn/scikit-learn/blob/1.0.2/sklearn/model_selection/_search.py#L493
        pytest.skip("Skipping tests that leverage passthrough to underlying estimator.")
    elif "nan" in check.func.__name__:
        pytest.skip(
            "Skipping tests that check for compatiblity with nulls. Underlying estimator should check."
        )
    else:
        check(estimator)


@pytest.mark.parametrize(
    "estimator,check",
    list(
        check_estimator(
            HyperoptSearchCV(
                estimator=RidgeClassifier(),
                space={"alpha": hp.uniform("alpha", 0, 1)},
                max_evals=10,
                random_state=42,
            ),
            generate_only=True,
        )
    ),
)
def test_estimator_classification(estimator, check):
    """Test compatibility with the scikit-learn API for classifiers."""
    if "predict" in check.func.__name__:
        # Predict methods do a simple passthrough to the underlying best estimator
        # https://github.com/scikit-learn/scikit-learn/blob/1.0.2/sklearn/model_selection/_search.py#L493
        pytest.skip("Skipping tests that leverage passthrough to underlying estimator.")
    elif "nan" in check.func.__name__:
        pytest.skip(
            "Skipping tests that check for compatiblity with nulls. Underlying estimator should check."
        )
    else:
        check(estimator)
