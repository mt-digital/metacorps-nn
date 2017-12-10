import sys

import pandas as pd

from modelrun import ModelRun


verbose = True
n_nodes = 500
# Keeping a ModelRun allows us to not have to re-load GoogleNews model.
rows = []  # Used to build data frame and latex table.
w2v_model_loc='GoogleNews-vectors-negative300.bin'

if len(sys.argv) > 1:
    run_directory = sys.argv[1]
    learning_rate = float(sys.argv[2])
else:
    run_directory = 'test_hyperparam_test'

mr = ModelRun(run_directory=run_directory,
              limit_word2vec=300000,
              w2v_model_loc=w2v_model_loc,
              learning_rate=learning_rate)

layer_sizes = [500, 300, 150, 100, 50]
for idx in range(1, len(layer_sizes) + 1):  # , n_hidden_layers in enumerate([1] + list(range(2, 9, 2))):

    # Parameter specifying number of 500-node hidden layers.
    # Confusing name, TODO change to hidden_layers all the way down.
    # mr.n_hidden = [n_nodes for _ in range(n_hidden_layers)]
    mr.n_hidden = layer_sizes[:idx]
    if verbose:
        print('Running with mr.n_hidden =', mr.n_hidden)
    ev = mr.run(n_epochs=120, verbose=verbose, early_stopping_limit=25)

    rows.append([
        'NN{}'.format(len(mr.n_hidden)),
        ev.sensitivity,
        ev.specificity,
        ev.precision,
        ev.auc
    ])

modelruns_table = pd.DataFrame(
    columns=['Method', 'Sens.', 'Spec.', 'Prec.', 'AUC'],
    data=rows
)
with open('PerformanceTable.tex', 'w') as f:
    modelruns_table.to_latex(f, float_format='%0.3f')
