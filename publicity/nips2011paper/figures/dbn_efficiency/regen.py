import sys
from hyperopt.icml07 import dataset_vsize_pairs
import matplotlib.pyplot as plt
import matplotlib
from hyperopt.rexp_plot import main_rexp_plot

# This script produces the figures for the DBN efficiency
# It brings in experimental results from a mongodb running on the local machine
# at the default port
#
# usage is  <script> <dataset> - to make a single figure
# or        <script>           - to make all the figures
#
#

matplotlib.rcParams['backend']='PDF'
matplotlib.rc('text', usetex=True)
matplotlib.rc('font',
        family='serif',
        serif='Computer Modern Roman',
        )
fontsize=16

if len(sys.argv)==2:
    #This part of the script is used for debugging,
    # to produce a single efficiency curve
    dataset = sys.argv[1]
    plt.figure(figsize=(4.3, 3.9))
    plt.axes([.19, .12, .75, .75])
    main_rexp_plot(
            'mongo://localhost/icml07/dbn_rand_jobs',
            dataset,
            post_filter=(lambda t:t['argd']['n_layers'] < 4),
            call_show=False,
            max_trials=256)
    #plt.legend(loc=4)
    plt.title(r'{\bf %s}'%''.join([(c if c != '_' else ' ') for c in dataset]),
            fontsize=fontsize)
    plt.ylabel(r'accuracy', fontsize=fontsize)
    plt.xlabel(r'experiment size (\# trials)', fontsize=fontsize)
    plt.savefig('dbn_efficiency_test.pdf')
else:
    for dataset, vsize in dataset_vsize_pairs:
        # render the raw preprocessing  figures
        plt.figure(figsize=(4.3,3.9))
        plt.axes([.19, .12, .75, .75])
        main_rexp_plot(
                'mongo://localhost/icml07/dbn_rand_jobs',
                dataset,
                query="{'argd.preprocessing':'raw'}",
                call_show=False,
                max_trials=None,
                hlines=['dbn1','dbn3'])
        plt.title(r'{\bf %s}'%''.join([(c if c != '_' else ' ') for c in
            dataset]), fontsize=fontsize)
        plt.ylabel(r'accuracy', fontsize=fontsize)
        plt.xlabel(r'experiment size (\# trials)', fontsize=fontsize)
        plt.savefig('dbn_efficiency_raw_%s.pdf'%dataset,)
        plt.cla()

        # render the full experiment figures
        plt.figure(figsize=(4.3,3.9))
        plt.axes([.19, .12, .75, .75])
        main_rexp_plot(
                'mongo://localhost/icml07/dbn_rand_jobs',
                dataset,
                query="{}",
                call_show=False,
                max_trials=None,
                hlines=['dbn1','dbn3'])
        plt.title(r'{\bf %s}'%''.join([(c if c != '_' else ' ') for c in
            dataset]), fontsize=fontsize)
        plt.ylabel(r'accuracy', fontsize=fontsize)
        plt.xlabel(r'experiment size (\# trials)', fontsize=fontsize)
        plt.savefig('dbn_efficiency_%s.pdf'%dataset,)
        plt.cla()

