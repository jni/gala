import os

from contextlib import contextmanager
from collections import OrderedDict

import numpy as np

from gala import imio, features, agglo, classify
from asv.extern.asizeof import asizeof


rundir = os.path.dirname(__file__)
## dd: the data directory
dd = os.path.abspath(os.path.join(rundir, '../tests/example-data'))


from time import process_time


@contextmanager
def timer():
    time = []
    t0 = process_time()
    yield time
    t1 = process_time()
    time.append(t1 - t0)


mh = features.default.moments_hist()


def trdata():
    wstr = imio.read_h5_stack(os.path.join(dd, 'train-ws.lzf.h5'))
    prtr = imio.read_h5_stack(os.path.join(dd, 'train-p1.lzf.h5'))
    gttr = imio.read_h5_stack(os.path.join(dd, 'train-gt.lzf.h5'))
    return wstr, prtr, gttr


def tsdata():
    wsts = imio.read_h5_stack(os.path.join(dd, 'test-ws.lzf.h5'))
    prts = imio.read_h5_stack(os.path.join(dd, 'test-p1.lzf.h5'))
    gtts = imio.read_h5_stack(os.path.join(dd, 'test-gt.lzf.h5'))
    return wsts, prts, gtts


def trgraph():
    ws, pr, ts = trdata()
    g = agglo.Rag(ws, pr)
    return g


def tsgraph():
    ws, pr, ts = tsdata()
    g = agglo.Rag(ws, pr, feature_manager=mh)
    return g


def trexamples():
    gt = imio.read_h5_stack(os.path.join(dd, 'train-gt.lzf.h5'))
    g = trgraph()
    (X, y, w, e), _ = g.learn_agglomerate(gt, mh, min_num_epochs=5)
    y = y[:, 0]
    return X, y


def classifier():
    X, y = trexamples()
    rf = classify.get_classifier('logistic')
    rf.fit(X, y)
    return rf


def policy():
    rf = classify.get_classifier('logistic')
    cl = agglo.classifier_probability(mh, rf)
    return cl


def tsgraph_queue():
    g = tsgraph()
    cl = policy()
    g.merge_priority_function = cl
    g.rebuild_merge_queue()
    return g

def bench_suite():
    times = OrderedDict()
    memory = OrderedDict()
    wstr, prtr, gttr = trdata()
    with timer() as t_build_rag:
        g = agglo.Rag(wstr, prtr)
    times['build RAG'] = t_build_rag[0]
    memory['base RAG'] = asizeof(g)
    with timer() as t_features:
        g.set_feature_manager(mh)
    times['build feature caches'] = t_features[0]
    memory['feature caches'] = asizeof(g) - memory['base RAG']
    with timer() as t_flat:
        _ignore = g.learn_flat(gttr, mh)
    times['learn flat'] = t_flat[0]
    with timer() as t_gala:
        (X, y, w, e), allepochs = g.learn_agglomerate(gttr, mh,
                                                      min_num_epochs=5,
                                                      classifier='logis')
        y = y[:, 0]  # ignore rand-sign and vi-sign schemes
    memory['training data'] = asizeof((X, y, w, e))
    times['learn agglo'] = t_gala[0]
    with timer() as t_train_classifier:
        cl = classify.get_classifier('logist')
        cl.fit(X, y)
    times['classifier training'] = t_train_classifier[0]
    memory['classifier training'] = asizeof(cl)
    policy = agglo.classifier_probability(mh, cl)
    wsts, prts, gtts = tsdata()
    gtest = agglo.Rag(wsts, prts, merge_priority_function=policy,
                      feature_manager=mh)
    with timer() as t_segment:
        gtest.agglomerate(np.inf)
    times['segment test volume'] = t_segment[0]
    memory['segment test volume'] = asizeof(gtest)
    return times, memory


def print_bench_results(times=None, memory=None):
    if times is not None:
        print('Timing results:')
        for key in times:
            print('--- ', key, times[key])
    if memory is not None:
        print('Memory results:')
        for key in memory:
            print('--- ', key, '%.3f MB' % (memory[key] / 1e6))


if __name__ == '__main__':
    times, memory = bench_suite()
    print_bench_results(times, memory)
