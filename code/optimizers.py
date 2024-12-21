import numpy as np
import tqdm
from importlib import reload

import gradient_approximation
import utils
reload(gradient_approximation)
reload(utils)

class GDOptimizer:
    def __init__(self, gradient_approximator, learning_rate_k, x_0,
                 x_sol=None, max_bits=10**4, tol=0.000001, seed=18):
        np.random.seed(seed)
        self.gradient_approximator = gradient_approximator # instnce *Aproximator
        self.learning_rate_k = learning_rate_k
        self.x_0 = x_0
        self.x_curr = np.copy(self.x_0)
        self.d = len(x_0)
        self.x_sol = x_sol
        self.func_name = gradient_approximator.func_name
        if self.func_name == "quadratic":
            self.A = gradient_approximator.A
            self.b = gradient_approximator.b
            self.c = gradient_approximator.c
        if self.func_name == "mushrooms":
            self.X = gradient_approximator.X
            self.y = gradient_approximator.y
            self.alpha = gradient_approximator.alpha

        if x_sol is not None:
            if self.func_name == "quadratic": 
                self.f_sol = utils.quadratic_func(self.x_sol, self.A, self.b, self.c)
            elif self.func_name == "mushrooms":
                self.f_sol = utils.logreg_func(self.x_sol, self.matrix, self.alpha)

        self.R0 = self.get_error(x_0)
        self.max_bits = max_bits
        self.tol = tol
        self.name = "GD"

    def step(self, x, k):
        gamma_k = self.learning_rate_k(k)
        nabla_f, num_bits = self.gradient_approximator.approx_gradient(x)
        x_next = x - gamma_k * nabla_f

        return x_next, num_bits
    
    def get_error(self, x):
        if self.x_sol is None: #||grad(x_k)||
            if self.func_name == "quadratic":
                error = np.linalg.norm(utils.quadratic_grad(x, self.A, self.b))
            elif self.func_name == "mushrooms":
                error = np.linalg.norm(utils.logreg_grad(x, self.X, self.y, alpha=self.alpha))
        else: #||x_k - x_sol||
            if self.func_name == "quadratic":
                error = np.linalg.norm(x - self.x_sol)
            if self.func_name == "mushrooms":
                error = np.linalg.norm(x - self.x_sol)
        return error
    
    def optimize(self):
        if self.gradient_approximator.name == "Bit norm grad":
            num_iter = self.max_bits // self.gradient_approximator.compressor.num_bits
        
        self.bits_list = [0]
        self.errors_list = [1.]
        for k in tqdm.trange(num_iter):
            self.x_curr, oracle_calls = self.step(self.x_curr, k)
            self.bits_list.append(self.bits_list[-1] + oracle_calls)
            error = self.get_error(self.x_curr) / self.R0
            self.errors_list.append(error)
            if error <= self.tol:
                print(f"Precision {self.tol} achieved at step {k}!")
                break
            if self.bits_list[-1] >= self.max_bits:
                break
