""" Misc Base classes
"""
import logging
logger = logging.getLogger(__name__)


class FakeCtrl(object):
    info = logger.info
    warn = logger.warn
    error = logger.error
    debug = logger.debug
    def partial_result(self, r):
        pass
    def checkpoint(self):
        pass


class SearchDomain(object):
    def __init__(self, template):
        self.template = template

    def dryrun_argd(self):
        """Return a point that could have been drawn from the template
        that is useful for small trial debugging.
        """
        raise NotImplementedError('override me')

    @classmethod
    def evaluate(cls, argd, ctrl):
        raise NotImplementedError('override me')

    @classmethod
    def main_dryrun(cls):
        self = cls()
        ctrl = FakeCtrl()
        argd = self.dryrun_argd()
        self.evaluate(argd, ctrl)

    if 0: #XXX decide whether to keep this - requires rewrite to work.
        @classmethod
        def main_plot_history(cls):
            import plotting
            conn_str = sys.argv[2]
            plotting.History(
                    MongoJobs.new_from_connection_str(conn_str),
                    trial_err_fn).main_show(title=conn_str)

        @classmethod
        def main_plot_histories(cls):
            import plotting
            conn_str_template = sys.argv[2]
            algos = sys.argv[3].split(',')
            dataset_name = sys.argv[4]
            start = int(sys.argv[5]) if len(sys.argv)>5 else 0
            stop = int(sys.argv[6]) if len(sys.argv)>6 else sys.maxint
            mh = plotting.MultiHistory()
            colors = ['r', 'y', 'b', 'g', 'c', 'k']


            def custom_err_fn(trial):
                if 2 == trial['status']:
                    rval = 1.0 - trial['result']['best_epoch_valid']
                    if rval > dict(
                            convex=.4,
                            mnist_rotated_background_images=2)[dataset_name]:
                        return None
                    else:
                        return rval

            for c, algo in zip(colors, algos):
                conn_str = conn_str_template % (algo, dataset_name)
                print 'algo', algo
                mh.add_experiment(
                        mj=MongoJobs.new_from_connection_str(conn_str),
                        y_fn=custom_err_fn,
                        color=c,
                        label=algo,
                        start=start,
                        stop=stop)
            plt = plotting.plt
            plt.axhline(
                    1.0 - icml07.dbn3_scores[dataset_name],
                    c='k',label='manual+grid')#, dashes=[0,1])
            mh.add_scatters()
            plt.legend()
            plt.title(dataset_name)
            plt.show()

        @classmethod
        def main_plot_scatter(cls):
            import plotting, ht_dist2
            conn_str = sys.argv[2]
            dataset_name = sys.argv[3]
            low_col = int(sys.argv[4])
            high_col = int(sys.argv[5])
            # upgrade jobs in db to ht_dist2-compatible things
            mj = MongoJobs.new_from_connection_str(conn_str)
            template = get_trial_template(dataset_name)
            trials = [job for job in mj]
            confs = [ht_dist2.bless(job['conf']) for job in trials]
            scatter_by_conf = plotting.ScatterByConf(
                    template['conf'],
                    confs,
                    status = [job['status'] for job in trials],
                    y = [trial_err_fn(job) for job in trials])
            sys.exit(scatter_by_conf.main_show_all(range(low_col, high_col)))
