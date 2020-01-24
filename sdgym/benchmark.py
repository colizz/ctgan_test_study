import logging

import pandas as pd

from sdgym.data import load_dataset
from sdgym.evaluate import evaluate

import pickle

LOGGER = logging.getLogger(__name__)


DEFAULT_DATASETS = [
    "adult",
##    "alarm",
##    "asia",
    "census",
##    "child",
    "covtype",
    "credit",
    "grid",
    "gridr",
##    "insurance",
##    "intrusion",
    "mnist12",
    "mnist28",
    "news",
    "ring"
]


def benchmark(synthesizer, datasets=DEFAULT_DATASETS, repeat=3, prefix='tmp'):
    print (datasets)
    results = list()
    for name in datasets:
        try:
            print('Evaluating dataset %s', name)
            train, test, meta, categoricals, ordinals = load_dataset(name, benchmark=True)

            for iteration in range(repeat):
                synthesized = synthesizer(train, categoricals, ordinals)
                scores = evaluate(train, test, synthesized, meta)
                scores['dataset'] = name
                scores['iter'] = iteration
                results.append(scores)
            print (results)
            with open(f'{prefix}_{name}.pickle', 'wb') as f:
                pickle.dump(results, f)
        except KeyError:
            print ("Here is the KeyError")
            continue

    return pd.concat(results)
