from tensorflow.contrib.opt import ExternalOptimizerInterface
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops import gradients
from tensorflow.python.ops import array_ops

#from numpy import *
#from math import *
#from numpy.linalg import *
import numpy as np

from optimizers import SplitExternalOptimizerInterface

""" Another attempt at external BFGS optimizer using contrib interface"""
class ExternalBFGSOptimizer(SplitExternalOptimizerInterface):

    @property
    def initialized(self):
        return self._initialized

    @initialized.setter
    def initialized(self, value):
        self._initialized = value

    def _initialize(self, x):
        N = np.shape(x)[0]
        self.H = 1.0 * np.eye(N)
        self.alpha = 1

    def _minimize(self, initial_val, loss_func, grad_func, equality_funcs,
                equality_grad_funcs, inequality_funcs, inequality_grad_funcs,
                step_callback, optimizer_kwargs):

        def unwrap_loss(x):
            return loss_func(x)[0]

        def unwrap_grad(x):
            return grad_func(x)[0]

        new_x = self.BFGS(unwrap_loss, unwrap_grad, initial_val)
        #self.x_last = initial_val
        return new_x

    def LineSearch(self, F, g, x, s, sigma=10 ** -1, beta=10, convergval=0.00001):
        # QFind returns 1 or the proper value (based on the current slope) of the line
        # based on basic rise over run of the distance of the current function with
        # new vs old value
        def QFind(alpha):
            if abs(alpha) < convergval:
                return 1
            return (F(x + alpha * s) - F(x)) / (alpha * np.dot(g, s))
        alpha = 1.
        # Double alpha until big enough
        while QFind(alpha) >= sigma:
            alpha = alpha * 2

        # BTacking
        while QFind(alpha) < sigma:
            alphap = alpha / (2.0 * (1 - QFind(alpha)))
            alpha = max(1.0 / beta * alpha, alphap)
        return alpha


    def BFGS(self, f, f_prime, x, eps=10e-8, tau=10e-6, sigma=10 ** -1, beta=10):
        # Startup
        if not self.initialized:
            self._initialize(x)

        g = f_prime(x)
        # move forward just one iteration, no tolerance or epsilon
        s = np.squeeze(-np.dot(self.H, g))
        # Repeating the linesearch
        alpha = self.LineSearch(f, g, x, s)
        x = x + alpha * s
        gold = g
        g = f_prime(x)
        y = (g - gold) / alpha
        dotsy = np.dot(s, y)
        if dotsy > 0:
            # Update H using estimation technique
            z = np.dot(self.H, y)
            self.H += np.outer(s, s) * (np.dot(s, y) + np.dot(y, z)) / dotsy ** 2 - (np.outer(z, s) + np.outer(s, z)) / dotsy
        return x