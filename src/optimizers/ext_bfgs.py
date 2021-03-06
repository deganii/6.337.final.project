import numpy as np

from optimizers import SplitExternalOptimizerInterface

""" An attempt at external BFGS optimizer using tensorflow contrib interface"""
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

    ''' the below method BFGS performs a *single* iteration of the BFGS method, and is called
         repeatedly by tensorflow.
    '''
    def BFGS(self, f, f_prime, w):
        if not self.initialized:
            self._initialize(w)
        fp_w = f_prime(w)
        dw = -np.dot(self.H, fp_w)
        alpha = self.line_search(f, fp_w, w, dw)
        w += alpha * dw
        fp_w_prev = fp_w
        fp_w = f_prime(w)
        y = (fp_w - fp_w_prev) / alpha
        sy = np.dot(dw, y)
        if sy > 0:
            # calculate an updated hessian approximation
            Hy = np.dot(self.H, y)
            term1 = np.outer(dw, dw) * (np.dot(dw, y) + np.dot(y, Hy)) / sy ** 2
            term2 = (np.outer(Hy, dw) + np.outer(dw, Hy)) / sy
            self.H += term1 - term2
        return w

    '''The "minimize" function is a standard interface required by tensorflow
       which we implement in order to call our custom BFGS optmimizer
    '''
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


    def Q(self, alpha, f, fp_w, w, s):
        tau = 0.00001
        if abs(alpha) < tau:
            return 1
        return (f(w + alpha * s) - f(w)) / (alpha * np.dot(fp_w, s))


    # Line search
    # https://en.wikipedia.org/wiki/Line_search
    def line_search(self, f, fp_w, w, s):
        c = 10 ** -1
        alpha = 1.
        while self.Q(alpha, f, fp_w, w, s) >= c:
            alpha = alpha * 2
        while self.Q(alpha, f, fp_w, w, s) < c:
            alpha_p = alpha / (2.0 * (1 - self.Q(alpha, f, fp_w, w, s)))
            alpha = max(1.0 / 10 * alpha, alpha_p)
        return alpha


