import scipy
from tensorflow.contrib.opt import ExternalOptimizerInterface
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops import gradients
from tensorflow.python.ops import array_ops

import numpy as np

from optimizers import *

""" Another attempt at external optimizer using contrib interface"""
class ExternalNewtonOptimizer(JacobianExternalOptimizerInterface):

  #def __init__(self, cost, learning_rate, name="ExternalPythonGradientDescentOptimizer"):
    """Construct a new gradient descent optimizer.
    Args:
      learning_rate: A Tensor or a floating point value.  The learning
        rate to use.
      use_locking: If True use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "GradientDescent".
    """
    #super(ExternalPythonGradientDescentOptimizer, self).__init__(cost )
    # self._learning_rate = learning_rate

    def _minimize(self, initial_val, loss_func, grad_func, jacob_func, equality_funcs,
                equality_grad_funcs, inequality_funcs, inequality_grad_funcs,
                step_callback, optimizer_kwargs):

        def unwrap_loss(x):
            return loss_func(x)[0]

        def unwrap_grad(x):
            return grad_func(x)[0]

        def unwrap_jacob(x):
            return jacob_func(x)[0]


        # turn the crank once
        fx =  unwrap_loss(initial_val)
        fpx =  unwrap_grad(initial_val)

        jacobian =   unwrap_jacob(initial_val)

        # need newx = initialval - inverse(Jacobain(intial_val) * F(initialval)

        inv_jac = np.linalg.inv(jacobian)

        new_x = initial_val - np.matmul(fx, inv_jac)

        # 1-d case
        #new_x = initial_val - fx / fpx

        return new_x
