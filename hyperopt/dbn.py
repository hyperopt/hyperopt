"""Deep Belief Network
"""
import copy
import cPickle
import logging
import os
import subprocess
import sys
import time
logger = logging.getLogger(__name__)

import numpy
from bson import SON, BSON

import theano
from theano import tensor

# scikits.data
from datasets.tasks import classification_train_valid_test
from datasets import dataset_factory

# XXX import source code into this project
from pylearn.shared.layers.logreg import LogisticRegression
import pylearn.gd.sgd

# XXX use scikits-learn?
import pylearn.preprocessing.pca

from base import SearchDomain

# XXX merge CURAND wrapper into Theano, use here.
RandomStreams = tensor.shared_randomstreams.RandomStreams

from ht_dist2 import rSON2, one_of, rlist, uniform, expon, geom, randint

# XXX: use scikits.data for this
def preprocess_data(argd, ctrl):
    dataset = dataset_factory(argd['dataset_name'])
    train, valid, test = classification_train_valid_test(dataset)
    X_train, y_train = numpy.asarray(train[0]), numpy.asarray(train[1])
    X_valid, y_valid = numpy.asarray(valid[0]), numpy.asarray(valid[1])
    X_test, y_test = numpy.asarray(test[0]), numpy.asarray(test[1])

    if argd['preprocessing']['kind'] == 'pca':
        # compute pca of input (TODO: retrieve only pca_whitened input)
        raise NotImplementedError('rewrite since cut and paste')
        (eigvals,eigvecs), centered_trainset = pylearn.preprocessing.pca.pca_from_examples(
                X=dataset['inputs'][:dataset['n_train']],
                max_energy_fraction=argd['pca_energy'])
        eigmean = dataset['inputs'][0] - centered_trainset[0]

        whitened_inputs = pylearn.preprocessing.pca.pca_whiten((eigvals,eigvecs),
                dataset['inputs']-eigmean)
        ctrl.info('PCA kept %i of %i components'%(whitened_inputs.shape[1],
            dataset['n_inputs']))
    elif argd['preprocessing']['kind'] == 'zca':
        (eigvals,eigvecs), centered_trainset = pylearn.preprocessing.pca.pca_from_examples(
                X=X_train,
                max_energy_fraction=argd['preprocessing']['energy'])
        eigmean = X_train[0] - centered_trainset[0]

        def whiten(X):
            X = pylearn.preprocessing.pca.pca_whiten((eigvals,eigvecs),
                    X - eigmean)
            X = pylearn.preprocessing.pca.pca_whiten_inverse((eigvals, eigvecs),
                    X) + eigmean
            X = X.astype('float32')
            X_min = X.min()
            X_max = X.max()
            ctrl.info('ZCA min:%f max:%f' % (X_min, X_max))
            if X_min < 0 or X_max > 1.0:
                ctrl.info('ZCA clamping return value to (0, 1) interval')
                X = numpy.clip(X, 0, 1, out=X)
            return X

        X_train, X_valid, X_test = [whiten(X)
                for X in [X_train, X_valid, X_test]]

    elif argd['preprocessing']['kind'] == 'normalize':
        raise NotImplementedError('rewrite since cut and paste')
        n_train=dataset['n_train']
        whitened_inputs = dataset['inputs']
        whitened_inputs = whitened_inputs - whitened_inputs[:n_train].mean(axis=0)
        whitened_inputs /= whitened_inputs[:n_train].std(axis=0)+1e-7
    elif argd['preprocessing']['kind'] == 'raw':
        pass
    else:
        raise ValueError(
                'unrecognized preprocessing',
                argd['preprocessing']['kind'])

    for Xy in 'X', 'y':
        for suffix in 'train', 'valid', 'test':
            varname = '%s_%s'%(Xy, suffix)
            var = locals()[varname]
            ctrl.info('%s shape=%s max=%f min=%f' % (
                varname,
                var.shape,
                var.max(),
                var.min()))

    s_X_train = theano.shared(X_train)
    s_y_train = theano.shared(y_train)
    s_X_valid = theano.shared(X_valid)
    s_y_valid = theano.shared(y_valid)
    s_X_test = theano.shared(X_test)
    s_y_test = theano.shared(y_test)

    return (dataset,
            (s_X_train, s_y_train),
            (s_X_valid, s_y_valid),
            (s_X_test, s_y_test))


def train_rbm(s_rng, s_idx, s_batchsize, s_features, W, vbias, hbias, n_in,
        n_hid, batchsize, sample_v0s,
        cdlr, n_epochs, n_batches_per_epoch, lr_anneal_start,
        givens={},
        time_limit=None):
    logger.info('rbm training n_in=%i n_hid=%i batchsize=%i' % (
        n_in, n_hid, batchsize))
    v0m = s_features
    if sample_v0s:
        v0s = tensor.cast(
                s_rng.uniform(size=(batchsize, n_in)) < v0m,
                'float32')
    else:
        v0s = v0m

    h0m = tensor.nnet.sigmoid(tensor.dot(v0s, W) + hbias)
    h0s = tensor.cast(s_rng.uniform(size=(batchsize, n_hid)) < h0m, 'float32')
    v1m = tensor.nnet.sigmoid(tensor.dot(h0s, W.T)+vbias)
    v1s = tensor.cast(s_rng.uniform(size=(batchsize, n_in)) < v1m, 'float32')
    h1m = tensor.nnet.sigmoid(tensor.dot(v1s, W) + hbias)

    s_lr = tensor.scalar(dtype='float32')

    logger.debug('compiling cd1_fn')
    cd1_fn = theano.function([s_idx, s_batchsize, s_lr],
            [abs(v0m-v1m).mean()],
            updates={
                W: W + s_lr * (
                    tensor.dot(v0s.T, h0m) - tensor.dot(v1s.T, h1m)),
                vbias: vbias + s_lr * (
                    (v0s - v1s).sum(axis=0)),
                hbias: hbias + s_lr * (
                    (h0m - h1m).sum(axis=0)),
                },
            givens=givens)
    for epoch in xrange(n_epochs):
        costs = []
        if time_limit and time.time() > time_limit:
            break
        e_lr = cdlr * min(1, (float(lr_anneal_start)/(epoch+1)))
        for batch_idx in xrange(n_batches_per_epoch):
            costs.append(cd1_fn(batch_idx, batchsize, e_lr))
        if not epoch % 10:
            logger.info('CD1 epoch:%i  avg L1: %f'% (epoch, numpy.mean(costs)))
    if costs:
        return dict(final_recon_l1=float(numpy.mean(costs)),)
    else:
        return dict(final_recon_l1=float('nan'))


class DBN_SearchDomain(SearchDomain):
    def __init__(self, dataset_name='rectangles', sup_min_epochs=300, sup_max_epochs=4000):
        template = rSON2(
            'preprocessing', one_of(
                rSON2(
                    'kind', 'raw'),
                rSON2(
                    'kind', 'zca',
                    'energy', uniform(0.5, 1.0))),
            'dataset_name', dataset_name,
            'sup_max_epochs', sup_max_epochs,
            'sup_min_epochs', sup_min_epochs,
            'iseed', one_of(5, 6, 7, 8),
            'batchsize', one_of(20, 100),
            'lr', expon(.001, 1),
            'lr_anneal_start', geom(100, 10000),
            'l2_penalty', one_of(0, expon(1e-7, 1e-4)),
            'next_layer', one_of(None,
                rSON2(
                    'n_hid', geom(2**7, 2**12, round=16),
                    'W_init_dist', one_of('uniform', 'normal'),
                    'W_init_algo', one_of('old', 'Xavier'),
                    'W_init_algo_old_multiplier', uniform(.2, 2),
                    'cd_epochs', geom(1, 3000),
                    'cd_batchsize', 100,
                    'cd_sample_v0s', one_of(False, True),
                    'cd_lr', expon(.0001, 1.0),
                    'cd_lr_anneal_start', geom(10, 10000),
                    'next_layer', one_of(None,
                        rSON2(
                            'n_hid', geom(2**7, 2**12, round=16),
                            'W_init_dist', one_of('uniform', 'normal'),
                            'W_init_algo', one_of('old', 'Xavier'),
                            'W_init_algo_old_multiplier', uniform(.2, 2),
                            'cd_epochs', geom(1, 2000),
                            'cd_batchsize', 100,
                            'cd_sample_v0s', one_of(False, True),
                            'cd_lr', expon(.0001, 1.0),
                            'cd_lr_anneal_start', geom(10, 10000),
                            'next_layer', one_of(None,
                                rSON2(
                                    'n_hid', geom(2**7, 2**12, round=16),
                                    'W_init_dist', one_of('uniform', 'normal'),
                                    'W_init_algo', one_of('old', 'Xavier'),
                                    'W_init_algo_old_multiplier', uniform(.2, 2),
                                    'cd_epochs', geom(1, 1500),
                                    'cd_batchsize', 100,
                                    'cd_sample_v0s', one_of(False, True),
                                    'cd_lr', expon(.0001, 1.0),
                                    'cd_lr_anneal_start', geom(10, 10000),
                                    'next_layer', None,
                                    )))))))
        SearchDomain.__init__(self, template)

    def dryrun_argd(self, *args, **kwargs):
        return dict(
                lr=.01,
                sup_max_epochs=500,
                sup_min_epochs=50,
                batchsize=10,
                preprocessing=dict(kind='zca', energy=0.8),
                iseed=5,
                n_layers=1,
                next_layer = dict(
                    n_hid=50,
                    W_init_dist='uniform',
                    W_init_algo='Xavier',
                    cd_epochs=100,
                    cd_batchsize=50,
                    cd_sample_v0s=True,
                    cd_lr=0.1,
                    cd_lr_anneal_start=3,
                    next_layer = dict(
                        n_hid=75,
                        W_init_dist='uniform',
                        W_init_algo='old',
                        W_init_algo_old_multiplier=2.2,
                        cd_epochs=70,
                        cd_batchsize=10,
                        cd_sample_v0s=False,
                        cd_lr=0.01,
                        cd_lr_anneal_start=30
                        ),
                    ),
                l2_penalty=0.1,
                lr_anneal_start=20,
                dataset_name='datasets.larochelle_etal_2007.Rectangles',
                )

    @classmethod
    def evaluate(cls, argd, ctrl):
        time_limit = time.time() + 60 * 60 # 1hr from now
        rval = SON(dbn_train_fn_version=1)

        ctrl.info('starting dbn_train_fn')
        kv = argd.items()
        kv.sort()
        for k,v in kv:
            ctrl.info('key=%s\t%s' %(k,str(v)))

        rng = numpy.random.RandomState(argd['iseed'])
        s_rng = RandomStreams(int(rng.randint(2**30)))

        dataset, train_Xy, valid_Xy, test_Xy = preprocess_data(argd, ctrl)

        # allocate learning function parameters
        s_inputs_all = tensor.fmatrix('inputs')
        s_labels_all = tensor.ivector('labels')
        s_idx = tensor.lscalar('batch_idx')
        s_batchsize=tensor.lscalar('batch_size')
        s_low = s_idx * s_batchsize
        s_high = s_low + s_batchsize
        s_inputs = s_inputs_all[s_low:s_high]
        s_labels = s_labels_all[s_low:s_high]
        s_lr = tensor.scalar('lr')
        s_features = s_inputs # s_features will be modified in the model-building loop

        weights = []
        vbiases = []
        hbiases = []

        n_inputs_i = valid_Xy[0].get_value(borrow=True).shape[1]

        rval['cd_reports'] = []

        layer_config = argd['next_layer']
        # allocate model parameters
        while layer_config:
            i = len(rval['cd_reports'])
            n_hid_i = layer_config['n_hid']
            if layer_config['W_init_dist']=='uniform':
                W = rng.uniform(low=-1,high=1,size=(n_hid_i, n_inputs_i)).T.astype('float32')
            elif layer_config['W_init_dist']=='normal':
                W = rng.randn(n_hid_i, n_inputs_i).T.astype('float32')
            else:
                raise ValueError('W_init_dist', layer_config['W_init_dist'])

            if layer_config['W_init_algo']=='old':
                #N.B. the weights are transposed so that as the number of hidden units changes,
                # the first hidden units are always the same vectors.
                # this makes it easier to isolate the effect of random initialization
                # from the other hyper-parameters under review
                W *= layer_config['W_init_algo_old_multiplier'] / numpy.sqrt(n_inputs_i)
            elif layer_config['W_init_algo']=='Xavier':
                W *= numpy.sqrt(6.0 / (n_inputs_i + n_hid_i))
            else:
                raise ValueError(layer_config['W_init_algo'])

            layer_idx = len(rval['cd_reports'])
            weights.append(theano.shared(W, 'W_%i' % layer_idx))
            hbiases.append(theano.shared(numpy.zeros(n_hid_i, dtype='float32'),
                'h_%i' % layer_idx))
            vbiases.append(theano.shared(numpy.zeros(n_inputs_i, dtype='float32'),
                'v_%i' % layer_idx))
            del W

            # allocate RBM training function for this layer
            # this version re-calculates the training set every time
            # TODO: cache the training set for each layer
            # TODO: consider sparsity?
            # TODO: consider momentum?
            if layer_config['cd_epochs']:
                cd_report = train_rbm(
                        s_rng, s_idx, s_batchsize, s_features,
                        W=weights[-1],
                        vbias=vbiases[-1],
                        hbias=hbiases[-1],
                        n_in=n_inputs_i,
                        n_hid=n_hid_i,
                        batchsize=layer_config['cd_batchsize'],
                        sample_v0s=layer_config['cd_sample_v0s'],
                        cdlr=layer_config['cd_lr'] / float(layer_config['cd_batchsize']),
                        n_epochs=layer_config['cd_epochs'],
                        n_batches_per_epoch=dataset.descr['n_train'] // layer_config['cd_batchsize'],
                        lr_anneal_start=layer_config['cd_lr_anneal_start'],
                        givens = {s_inputs_all: train_Xy[0]},
                        time_limit=time_limit
                        )
            else:
                cd_report = None
            rval['cd_reports'].append(cd_report)

            # update s_features to point to top layer
            s_features = tensor.nnet.sigmoid(
                    tensor.dot(s_features, weights[-1]) + hbiases[-1])
            n_inputs_i = n_hid_i
            layer_config = layer_config.get('next_layer', None)

        # allocate model

        logreg = LogisticRegression.new(s_features, n_in=n_inputs_i,
                n_out=dataset.descr['n_classes'])
        traincost = logreg.nll(s_labels).mean()
        def ssq(X):
            return (X**2).sum()
        traincost = traincost + argd['l2_penalty'] * (
                sum([ssq(w_i) for w_i in weights]) + ssq(logreg.w))
        # params = weights+hbiases+vbiases+logreg.params
        # vbiases are not involved in the supervised network
        params = weights + hbiases + logreg.params
        train_logreg_fn = theano.function([s_idx, s_lr],
                [logreg.nll(s_labels).mean()],
                updates=pylearn.gd.sgd.sgd_updates(
                    params=params,
                    grads=tensor.grad(traincost, params),
                    stepsizes=[s_lr] * len(params)),
                givens={s_batchsize:argd['batchsize'],
                    s_inputs_all: train_Xy[0],
                    s_labels_all: train_Xy[1]})
        valid_logreg_fn = theano.function([s_idx],
            logreg.errors(s_labels).mean(),
            givens={s_batchsize:argd['batchsize'],
                s_inputs_all: valid_Xy[0],
                s_labels_all: valid_Xy[1]})
        test_logreg_fn = theano.function([s_idx],
            logreg.errors(s_labels).mean(),
            givens={s_batchsize:argd['batchsize'],
                s_inputs_all: test_Xy[0],
                s_labels_all: test_Xy[1]})

        rval['best_epoch'] = -1
        rval['best_epoch_valid'] = -1
        rval['best_epoch_train'] = -1
        rval['best_epoch_test'] = -1
        valid_rate=-1
        test_rate=-1
        train_rate=-1

        n_train_batches = dataset.descr['n_train'] // argd['batchsize']
        n_valid_batches = dataset.descr['n_valid'] // argd['batchsize']
        n_test_batches = dataset.descr['n_test'] // argd['batchsize']

        n_iters = 0
        for epoch in xrange(argd['sup_max_epochs']):
            e_lr = argd['lr']
            e_lr *= min(1, argd['lr_anneal_start'] / float(n_iters+1)) #anneal learning rate
            valid_rate = float(1 - numpy.mean([valid_logreg_fn(i)
                for i in range(n_valid_batches)]))
            valid_rate_std_thresh = 0.5 * numpy.sqrt(valid_rate *
                    (1 - valid_rate) / (n_valid_batches * argd['batchsize']))

            if valid_rate > (rval['best_epoch_valid']+valid_rate_std_thresh):
                rval['best_epoch'] = epoch
                rval['best_epoch_test'] = test_rate
                rval['best_epoch_valid'] = valid_rate
                rval['best_epoch_train'] = train_rate
                best_params = copy.deepcopy(params)
            logger.info('Epoch=%i best epoch %i valid %f test %f best_epoch_train %f prev_train %f'%(
                epoch, rval['best_epoch'], rval['best_epoch_valid'], rval['best_epoch_test'],
                    rval['best_epoch_train'], train_rate))
            #ctrl.info('Epoch %i train nll: %f'%(epoch, train_rate))
            ctrl.partial_result(rval)
            ctrl.checkpoint()

            if epoch > argd['sup_min_epochs'] and epoch > 2*rval['best_epoch']:
                break
            if time.time() > time_limit:
                break
            train_rate = float(numpy.mean([train_logreg_fn(i,e_lr) for i in
                range(n_train_batches)]))
            ++n_iters

        do_test=1
        if do_test:
            # copy best params back into place
            for p, bp in zip(params, best_params):
                p.set_value(bp.get_value())
            rval['best_epoch_test'] = 1-float(
                    numpy.mean(
                        [test_logreg_fn(i) for i in range(n_test_batches)]))
        return rval

