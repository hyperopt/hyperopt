# Not unit testing, just some random small tests.

from sklearn import datasets
from hyperopt import hp

iris = datasets.load_iris()
X, y = iris.data, iris.target
X, y = X[y != 0, :2], y[y != 0]
X_og, y_og = X, y

space = hp.pchoice('naive_type', [.14, .02, .84],
                   ['gaussian', 'multinomial', 'bernoulli'])
import hyperopt.pyll.stochastic
a, b, c = 0, 0, 0
for i in range(0, 1000):
    nesto = hyperopt.pyll.stochastic.sample(space)
    if nesto == 'gaussian':
        a += 1
    elif nesto == 'multinomial':
        b += 1
    else:
        c += 1
print(a, b, c)


space = hp.choice('normal_choice', [
    hp.pchoice('fsd', [.1, .8, .1], ['first', 'second', 2]),
    hp.choice('something_else', [10, 20])
])
a, b, c = 0, 0, 0
for i in range(0, 1000):
    nesto = hyperopt.pyll.stochastic.sample(space)
    if nesto == 'first':
        a += 1
    elif nesto == 'second':
        b += 1
    elif nesto == 2:
        c += 1
print(a, b, c)
