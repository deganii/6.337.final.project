"""GradientDescent for TensorFlow, originally obtained from:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/gradient_descent.py

Now rewritten purely in python using C++ guideline:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/training_ops.py
rewritten in pure python."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops


class PurePythonGradientDescentOptimizer(optimizer.Optimizer):
  """Optimizer that implements the gradient descent algorithm, purely in python
  """


  def __init__(self, learning_rate, use_locking=False, name="GradientDescent"):
    """Construct a new gradient descent optimizer.
    Args:
      learning_rate: A Tensor or a floating point value.  The learning
        rate to use.
      use_locking: If True use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "GradientDescent".
    """
    super(PurePythonGradientDescentOptimizer, self).__init__(use_locking, name)
    self._learning_rate = learning_rate

  def _apply_dense(self, grad, var):
    d = math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype)
    training_res = training_ops.apply_gradient_descent(
        var,
        math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
        grad,
        use_locking=self._use_locking)
    training_res_op = training_res.op
    var = var - self._learning_rate_tensor * grad
    return var.op

  def _resource_apply_dense(self, grad, handle):
    training_res = training_ops.resource_apply_gradient_descent(
        handle.handle, math_ops.cast(self._learning_rate_tensor,
                                     grad.dtype.base_dtype),
        grad, use_locking=self._use_locking)
    return training_res

  def _resource_apply_sparse_duplicate_indices(self, grad, handle, indices):
    return resource_variable_ops.resource_scatter_add(
        handle.handle, indices, -grad * self._learning_rate)

  def _apply_sparse_duplicate_indices(self, grad, var):
    delta = ops.IndexedSlices(
        grad.values *
        math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
        grad.indices, grad.dense_shape)
    return var.scatter_sub(delta, use_locking=self._use_locking)

  def _apply_sparse(self, grad, var):
      pass

  def _prepare(self):
    self._learning_rate_tensor = ops.convert_to_tensor(self._learning_rate,
                                                       name="learning_rate")