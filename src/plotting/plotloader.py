import numpy as np
import numpy as np
import glob
from plotting import PerformancePlotter
import ntpath
# load all from a specific directory

import os

pretty_print = {
    'ext_grad' : 'Gradient Descent',
    'adam': 'Adam Optimizer',
    'ext_bfgs': 'BFGS'
}

datasets = {}
dir = "../performance_data/toy/*.npz"
for file in glob.glob(dir):
    npzfile = np.load(file)
    optim = ntpath.basename(os.path.splitext(file)[0])
    datasets[pretty_print[optim]] = npzfile

# plot the performance
optims = list(datasets.keys())

PerformancePlotter.plot_loss_multi(datasets)

