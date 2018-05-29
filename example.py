from hyperopt import hp
from hyperopt.suggest import suggest, dataframe_to_trials
from pandas import DataFrame
import numpy as np
import argparse
import matplotlib.pyplot as plt


test_config = dict(
    univariate_simple=dict(
        # 2x + 1
        # min val = -199, x = -199
        func=lambda input_: 2 * input_[0] + 1,
        space=[hp.uniform('x', -100, 100)]
    ),
    univariate=dict(
        # x^3 - 2x^2 + sqrt(x)
        # min val = -0.0566466, x = 1.20777
        func=lambda input_: input_[0]**3 - 2*input_[0]**2 + np.sqrt(input_[0]),
        space=[hp.uniform('x', 0, 100)]
    ),
    multivariate=dict(
        # 2x^2 + xy + y^2
        # simple function to test
        # min val = 0, x = 0, y = 0
        func=lambda input_: 2 * input_[0] ** 2 + input_[0] * input_[1] + input_[1]**2,
        space=[hp.uniform('x', 0, 100), hp.uniform('y', 0, 100)]
    )
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', default='multivariate', type=str)
    parser.add_argument('--num_iter', default=20, type=int)
    parser.add_argument('--num_trials', default=3, type=int)
    args = parser.parse_args()

    data = DataFrame()
    config = test_config[args.test]
    func = config['func']
    space = config['space']

    plot_data = []
    for _ in range(args.num_iter):
        data = suggest(data, space, num_trials=args.num_trials)
        for index in [-j for j in range(args.num_trials)]:
            index = len(data) - 1 + index
            data.set_value(index, 'loss', func(data.iloc[index].tolist()))
        trials = dataframe_to_trials(data)
        trials.refresh()
        argmin = trials.argmin
        min_loss = func(list(argmin.values()))
        print('iter={}, argmin={}, loss={}'.format(_, argmin, min_loss))
        plot_data.append(min_loss)

    print('Trials data: ')
    print(data)
    plt.figure()
    plt.loglog(plot_data)
    plt.show()
