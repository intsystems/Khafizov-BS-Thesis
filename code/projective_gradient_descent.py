import numpy as np

class GradientDescentOnCube:
    def __init__(self, w_0, grad, gamma_0, gamma=0.1):
        self.w = w_0
        self.grad = grad
        self.gamma = gamma
        self.gamma_0 = gamma_0
    
    def fit(self, x_k, max_iter=20):
        g_k = self.grad(x_k)
        for i in range(max_iter):
            g = -self.grad(x_k - self.gamma_0 * self.w * g_k) * self.gamma_0 * g_k
            self.w -= self.gamma * g
            self.w = np.maximum(0, np.minimum(1, self.w))
