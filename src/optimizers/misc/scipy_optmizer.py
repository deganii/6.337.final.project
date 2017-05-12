
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib.opt import ExternalOptimizerInterface
import scipy.optimize


class ScipyOptimizerInterface(ExternalOptimizerInterface):
  #_DEFAULT_METHOD = 'Nelder-Mead'
  #_DEFAULT_METHOD = 'Powell'
  #_DEFAULT_METHOD = 'L-BFGS-B'
  #_DEFAULT_METHOD = 'BFGS'
  #_DEFAULT_METHOD = 'trust-ncg'
  #_DEFAULT_METHOD = 'dogleg'
  #_DEFAULT_METHOD = 'CG' # conjugate gradient
  _DEFAULT_METHOD = 'Newton-CG'
  #_DEFAULT_METHOD = 'TNC' #  truncated Newton
  #_DEFAULT_METHOD = 'COBYLA' # Constrained Optimization BY Linear Approximation

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

      # pylint: disable=g-import-not-at-top
    result = scipy.optimize.minimize(*minimize_args, **minimize_kwargs)
    logging.info('Optimization terminated with:\n'
                 '  Message: %s\n'
                 '  Objective function value: %f\n'
                 '  Number of iterations: %d\n'
                 '  Number of functions evaluations: %d',
                 result.message, result.fun, result.nit, result.nfev)

    return result['x']