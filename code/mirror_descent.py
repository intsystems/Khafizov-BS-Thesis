import numpy as np

class MirrorDescentOnSimplex:
    def __init__(self, w_0, grad, alpha, gamma_0, gamma=0.1):
        self.w = w_0
        self.grad = grad
        self.reg_grad = lambda w: np.log(w)
        self.alpha = alpha
        self.gamma = gamma
        self.gamma_0 = gamma_0
    
    def fit(self, x_k, max_iter=20):
        g_k = self.grad(x_k)
        for i in range(max_iter):
            g = -self.grad(x_k - self.gamma_0 * self.w / np.max(self.w) * g_k) * self.gamma_0 * g_k + self.alpha * self.reg_grad(self.w)
            self.w = self.w * np.exp(-self.gamma * g)
            self.w /= np.linalg.norm(self.w, ord=1)
