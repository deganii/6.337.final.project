import scipy
from tensorflow.contrib.opt import ExternalOptimizerInterface
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops import gradients
from tensorflow.python.ops import array_ops

""" Another attempt at external optimizer using contrib interface"""
class ExternalPythonGradientDescentOptimizer(ExternalOptimizerInterface):

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self._learning_rate = value

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

    def _minimize(self, initial_val, loss_grad_func, equality_funcs,
                equality_grad_funcs, inequality_funcs, inequality_grad_funcs,
                step_callback, optimizer_kwargs):
        _, grad = loss_grad_func(initial_val)
        return initial_val - self._learning_rate*grad

