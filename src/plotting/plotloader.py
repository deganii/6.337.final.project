import numpy as np


import numpy as np
from plotting import PerformancePlotter

# load all from a specific directory

import os

pretty_print = {
    'ext_grad' : 'Gradient Descent',
    'adam': 'Adam Optimizer',
    'ext_bfgs': 'BFGS'
}

datasets = {}
dir = "../performance_data/toy/"
for file in os.listdir(dir):
    npzfile = np.load(os.path.join(dir,file))
    optim = os.path.splitext(file)[0]
    datasets[pretty_print[optim]] = npzfile

# plot the performance
optims = list(datasets.keys())

PerformancePlotter.plot_loss_multi(datasets)

