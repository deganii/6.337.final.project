import scipy
from tensorflow.contrib.opt import ExternalOptimizerInterface
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops import gradients
from tensorflow.python.ops import array_ops

""" Another attempt at external optimizer using contrib interface"""
class ExternalPythonGradientDescentOptimizer(ExternalOptimizerInterface):
  """Wrapper allowing `scipy.optimize.minimize` to operate a `tf.Session`.
  Example:
  ```python
  vector = tf.Variable([7., 7.], 'vector')
  # Make vector norm as small as possible.
  loss = tf.reduce_sum(tf.square(vector))
  optimizer = ScipyOptimizerInterface(loss, options={'maxiter': 100})
  with tf.Session() as session:
    optimizer.minimize(session)
  # The value of vector should now be [0., 0.].
  ```
  Example with constraints:
  ```python
  vector = tf.Variable([7., 7.], 'vector')
  # Make vector norm as small as possible.
  loss = tf.reduce_sum(tf.square(vector))
  # Ensure the vector's y component is = 1.
  equalities = [vector[1] - 1.]
  # Ensure the vector's x component is >= 1.
  inequalities = [vector[0] - 1.]
  # Our default SciPy optimization algorithm, L-BFGS-B, does not support
  # general constraints. Thus we use SLSQP instead.
  optimizer = ScipyOptimizerInterface(
      loss, equalities=equalities, inequalities=inequalities, method='SLSQP')
  with tf.Session() as session:
    optimizer.minimize(session)
  # The value of vector should now be [1., 1.].
  ```
  """

  _DEFAULT_METHOD = 'L-BFGS-B'

  def _minimize(self, initial_val, loss_grad_func, equality_funcs,
                equality_grad_funcs, inequality_funcs, inequality_grad_funcs,
                step_callback, optimizer_kwargs):
    def loss_grad_func_wrapper(x):
      # SciPy's L-BFGS-B Fortran implementation requires gradients as doubles.
      loss, gradient = loss_grad_func(x)
      return loss, gradient.astype('float64')

    method = optimizer_kwargs.pop('method', self._DEFAULT_METHOD)

    constraints = []
    for func, grad_func in zip(equality_funcs, equality_grad_funcs):
      constraints.append({'type': 'eq', 'fun': func, 'jac': grad_func})
    for func, grad_func in zip(inequality_funcs, inequality_grad_funcs):
      constraints.append({'type': 'ineq', 'fun': func, 'jac': grad_func})

    minimize_args = [loss_grad_func_wrapper, initial_val]
    minimize_kwargs = {
        'jac': True,
        'callback': step_callback,
        'method': method,
        'constraints': constraints,
    }
    minimize_kwargs.update(optimizer_kwargs)
    if method == 'SLSQP':
      # SLSQP doesn't support step callbacks. Obviate associated warning
      # message.
      del minimize_kwargs['callback']

    import scipy.optimize  # pylint: disable=g-import-not-at-top
    result = scipy.optimize.minimize(*minimize_args, **minimize_kwargs)
    logging.info('Optimization terminated with:\n'
                 '  Message: %s\n'
                 '  Objective function value: %f\n'
                 '  Number of iterations: %d\n'
                 '  Number of functions evaluations: %d',
                 result.message, result.fun, result.nit, result.nfev)

    return result['x']


def _accumulate(list_):
  total = 0
  yield total
  for x in list_:
    total += x
    yield total


def _get_shape_tuple(tensor):
  return tuple(dim.value for dim in tensor.get_shape())


def _prod(array):
  prod = 1
  for value in array:
    prod *= value
  return prod


def _compute_gradients(tensor, var_list):
  grads = gradients.gradients(tensor, var_list)
  # tf.gradients sometimes returns `None` when it should return 0.
  return [grad if grad is not None else array_ops.zeros_like(var)
          for var, grad in zip(var_list, grads)]