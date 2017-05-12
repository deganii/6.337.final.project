from tensorflow.contrib.opt import ExternalOptimizerInterface
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops import gradients
from tensorflow.python.ops import array_ops

from numpy import *
from math import *
from numpy.linalg import *

from optimizers import SplitExternalOptimizerInterface

""" Another attempt at external optimizer using contrib interface"""
class ExternalDownhillSimplexOptimizer(SplitExternalOptimizerInterface):

    @property
    def initialized(self):
        return self._initialized

    @initialized.setter
    def initialized(self, value):
        self._initialized = value

    def _initialize(self, x):
        N = shape(x)[0]
        self.H = 1.0 * eye(N)
        self.alpha = 1

    def _minimize(self, initial_val, loss_func, grad_func, equality_funcs,
                equality_grad_funcs, inequality_funcs, inequality_grad_funcs,
                step_callback, optimizer_kwargs):

        def unwrap_loss(x):
            return loss_func(x)[0]

        def unwrap_grad(x):
            return grad_func(x)[0]

        new_x,counter = self.DownhillSimplex(unwrap_loss, initial_val, slide=0.0005, tol = 1.0 ** -3)
        #self.x_last = initial_val
        return new_x


    def DownhillSimplex(self, F, Start, slide=1.0, tol=1.0 ** -8):
        # Setup intial values

        n = len(Start)
        f = zeros(n + 1)
        x = zeros((n + 1, n))

        x[0] = Start

        # Setup intial X range

        for i in range(1, n + 1):
            x[i] = Start
            x[i, i - 1] = Start[i - 1] + slide

        # Setup intial functions based on x's just defined

        for i in range(n + 1):
            f[i] = F(x[i])

        # Main Loop operation, loops infinitely until break condition
        counter = 0


        while True:

            low = argmin(f)
            high = argmax(f)

            # Implement Counter
            counter += 1

            # Compute Migration
            d = (-(n + 1) * x[high] + sum(x)) / n

            if sqrt(dot(d, d) / n) < tol:
                # Break condition, value is darn close
                return (x[low], counter)

            newX = x[high] + 2.0 * d
            newF = F(newX)

            # Bad news, new values is lower than p. low

            if newF <= f[low]:
                x[high] = newX
                f[high] = newF
                newX = x[high] + d
                newF = F(newX)
                # Maybe I need to expand
                if newF <= f[low]:
                    x[high] = newX
                    f[high] = newF
            # Good news, new values is higher
            else:
                # Do I need to contract?
                if newF <= f[high]:
                    x[high] = newX
                    f[high] = newF
                else:
                    # Contraction
                    newX = x[high] + 0.5 * d
                    newF = F(newX)
                    if newF <= f[high]:
                        x[high] = newX
                        f[high] = newF
                    else:
                        for i in range(len(x)):
                            if i != low:
                                x[i] = (x[i] - x[low])
                                f[i] = F(x[i])
