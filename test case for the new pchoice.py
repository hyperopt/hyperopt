from hyperopt import fmin, tpe, Trials

from hyperopt import hp
space = hp.pchoice('something', [
    (.2, hp.pchoice('number', [(.8, 2), (.2, 1)])),
    (.8, hp.pchoice('number1', [(.7, 5), (.3, 6)]))])

best = fmin(fn=lambda x: x,
            space=space,
            algo=tpe.suggest,
            max_evals=50,
            trials=Trials())
