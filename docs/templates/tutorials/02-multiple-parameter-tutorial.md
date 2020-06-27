# 02. MultipleParameterTutorial

In [01.BasicTutorial](https://github.com/hyperopt/hyperopt/blob/master/tutorial/01.BasicTutorial.ipynb), you learned about optimizing a single HyperParameter function. At MultipleParameterTurial, you learn the following things.

* Optimize the Objective Function with Multiple HyperParameters
* Define various search space


```python
# Import HyperOpt Library
from hyperopt import tpe, hp, fmin
import numpy as np
```

Declares a purpose function to optimize. Unlike last time, we will optimize the function with two Hyperparameters, $x$ and $y$.

$$ z = sin\sqrt{x^2 + y^2} $$


```python
def objective(params):
    x, y = params['x'], params['y']
    return np.sin(np.sqrt(x**2 + y**2))
```

Just like last time, let's try visualizing it. But unlike last time, there are two Hyperparameters, so we need to visualize them in 3D space.


```python
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)
x, y = np.meshgrid(x, y)

z = objective({'x': x, 'y': y})

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(x, y, z, cmap=cm.coolwarm)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()
```


![png](02-multiple-parameter-tutorial_files/02-multiple-parameter-tutorial_5_0.png)


Likewise, let's define the search space. However, this time, you need to define two search spaces($x, y$), so you put each of them in the `dict()`.


```python
space = {
    'x': hp.uniform('x', -6, 6),
    'y': hp.uniform('y', -6, 6)
}
```

Perfect! Now you can do exactly what you did at BasicTutorial!


```python
best = fmin(
    fn=objective, # Objective Function to optimize
    space=space, # Hyperparameter's Search Space
    algo=tpe.suggest, # Optimization algorithm (representative TPE)
    max_evals=1000 # Number of optimization attempts
)
print(best)
```

    100%|██████████| 1000/1000 [00:08<00:00, 124.16trial/s, best loss: -0.9999998753676712]
    {'x': 4.283678790265057, 'y': 1.9626514514966948}


## Using various search spaces 

`hp.randint(label, upper)` searches the integer in the [0, upper) interval.


```python
f = lambda x: -x

best = fmin(
    fn=f,
    space=hp.randint('x', 5),
    algo=tpe.suggest,
    max_evals=10
)
print(best)
```

    100%|██████████| 10/10 [00:00<00:00, 877.51trial/s, best loss: -4.0]
    {'x': 4}


`hp.choice(label, list)` searches for elements in the list.


```python
def f(x):
    if x == 'james':
        return 0
    if x == 'max':
        return 1
    if x == 'wansoo':
        return 2

best = fmin(
    fn=f,
    space=hp.choice('x', ['james', 'max', 'wansoo']),
    algo=tpe.suggest,
    max_evals=10
)
print(best)
```

    100%|██████████| 10/10 [00:00<00:00, 638.73trial/s, best loss: 0.0]
    {'x': 0}

