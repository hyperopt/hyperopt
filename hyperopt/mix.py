from builtins import zip
import numpy as np


def suggest(new_ids, domain, trials, seed, p_suggest):
    """Return the result of a randomly-chosen suggest function

    For exampl to search by sometimes using random search, sometimes anneal,
    and sometimes tpe, type:

        fmin(...,
            algo=partial(mix.suggest,
                p_suggest=[
                    (.1, rand.suggest),
                    (.2, anneal.suggest),
                    (.7, tpe.suggest),]),
            )


    Parameters
    ----------

    p_suggest: list of (probability, suggest) pairs
        Make a suggestion from one of the suggest functions,
        in proportion to its corresponding probability.
        sum(probabilities) must be [close to] 1.0

    """
    rng = np.random.RandomState(seed)
    ps, suggests = list(zip(*p_suggest))
    assert len(ps) == len(suggests) == len(p_suggest)
    if not np.isclose(sum(ps), 1.0):
        raise ValueError('Probabilities should sum to 1', ps)
    idx = rng.multinomial(n=1, pvals=ps).argmax()
    return suggests[idx](new_ids, domain, trials,
                         seed=int(rng.randint(2 ** 31)))
