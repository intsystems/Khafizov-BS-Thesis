import numpy as np
from importlib import reload

import utils
from mirror_descent import MirrorDescentOnSimplex
from projective_gradient_descent import GradientDescentOnCube
reload(utils)

def greedy_compress(g, indices):
    x = g[indices]

    zero_indices = (x == 0)
    nonzero_indices = (x != 0)

    y = x[nonzero_indices]

    power = np.maximum(-np.round(np.log2(np.abs(y))), np.zeros_like(y))
    bits = np.ceil(np.log2(power + 1))

    y = np.sign(y) / (2**power)

    x[nonzero_indices] = y

    result = np.zeros_like(g)
    result[indices] = x

    return result, len(x) + np.sum(bits - 1)



# class MirrorDescentCompressor:
#     def __init__(self, grad, gamma, shape, use_ratio=0.1):
#         self.grad = grad
#         self.gamma = gamma
#         self.use_ratio = use_ratio
#         self.num_bits = int((use_ratio * 3 + 1) * shape) # approximate amount of bits
#         self.w = np.ones(shape=shape)
#         self.w /= np.linalg.norm(self.w, ord=1)
    
#     def __call__(self, x):
#         x = np.copy(x)

#         MD = MirrorDescentOnSimplex(self.w, self.grad, 0.1, gamma_0=self.gamma, gamma=1.)

#         MD.fit(x, max_iter=100)
#         self.w = MD.w
        
#         g_k = self.grad(x)
#         indices = np.random.choice(len(self.w), size=int(len(self.w) * self.use_ratio), replace=False, p=self.w)
        
#         result, self.num_bits = greedy_compress(g_k, indices)

#         return result

class MirrorDescentGreedyCompressor:
    def __init__(self, grad, gamma, shape, use_ratio=0.1):
        self.grad = grad
        self.gamma = gamma
        self.use_ratio = use_ratio
        self.num_bits = int((use_ratio * 3 + 1) * shape)
        self.w = np.ones(shape=shape)
    
    def __call__(self, x):
        x = np.copy(x)

        k = int(len(x) * self.use_ratio)
        d = len(x)

        MD = MirrorDescentOnSimplex(self.w, self.grad, 0.1, gamma_0=self.gamma, gamma=1., R=d)

        MD.fit(x, max_iter=100)
        self.w = MD.w
        
        g_k = self.grad(x)
        indices = np.argsort(-self.w)[:k]
        
        result, self.num_bits = greedy_compress(g_k, indices)

        return result


class MirrorDescentGreedyWeightedCompressor:
    def __init__(self, grad, gamma, shape, use_ratio=0.1):
        self.grad = grad
        self.gamma = gamma
        self.use_ratio = use_ratio
        self.num_bits = int((use_ratio * 3 + 1) * shape)
        self.w = np.ones(shape=shape)
    
    def __call__(self, x):
        x = np.copy(x)

        k = int(len(x) * self.use_ratio)
        d = len(x)

        MD = MirrorDescentOnSimplex(self.w, self.grad, 0.1, gamma_0=self.gamma, gamma=1., R=d)

        MD.fit(x, max_iter=100)
        self.w = MD.w
        
        g_k = self.grad(x)
        indices = np.argsort(-self.w)[:k]
        
        weights = np.zeros_like(g_k)
        weights[indices] = self.w[indices]
        weights *= k / weights.sum()
        
        result, self.num_bits = greedy_compress(weights * g_k, indices)

        return result
    
class MirrorDescentWeightedTopkCompressor:
    def __init__(self, grad, gamma, shape, use_ratio=0.1):
        self.grad = grad
        self.gamma = gamma
        self.use_ratio = use_ratio
        self.num_bits = int((use_ratio * 3 + 1) * shape)
        self.w = np.ones(shape=shape)
    
    def __call__(self, x):
        x = np.copy(x)

        k = int(len(x) * self.use_ratio)
        d = len(x)

        MD = MirrorDescentOnSimplex(self.w, self.grad, 0.1, gamma_0=self.gamma, gamma=1., R=d)

        MD.fit(x, max_iter=100)
        self.w = MD.w
        
        g_k = self.grad(x)
        indices = np.argsort(-self.w * np.abs(g_k))[:k]
        
        weights = np.zeros_like(g_k)
        weights[indices] = self.w[indices]
        weights *= k / weights.sum()
        
        result, self.num_bits = greedy_compress(weights * g_k, indices)

        return result



class SquareGradientCompressor:
    def __init__(self, grad, shape, use_ratio=0.1):
        self.grad = grad
        self.use_ratio = use_ratio
        self.num_bits = int((use_ratio * 3 + 1) * shape)
    
    def __call__(self, x):
        x = np.copy(x)
        
        g_k = self.grad(x)
        w = g_k**2
        w /= np.linalg.norm(w, ord=1)
        indices = np.random.choice(len(g_k), size=int(len(g_k) * self.use_ratio), replace=False, p=w)

        result, self.num_bits = greedy_compress(g_k, indices)

        return result
    
class GradientDescentGreedyCompressor:
    def __init__(self, grad, gamma, shape, use_ratio=0.1):
        self.grad = grad
        self.gamma = gamma
        self.use_ratio = use_ratio
        self.num_bits = int((use_ratio * 3 + 1) * shape)
        self.w = np.ones(shape=shape) / 2
    
    def __call__(self, x):
        x = np.copy(x)

        k = int(len(x) * self.use_ratio)
        d = len(x)

        GD = GradientDescentOnCube(self.w, self.grad, gamma_0=self.gamma, gamma=1.)

        GD.fit(x, max_iter=100)
        self.w = GD.w
        
        g_k = self.grad(x)
        indices = np.argsort(-self.w)[:k]
        
        result, self.num_bits = greedy_compress(g_k, indices)

        return result

class GradientDescentGreedyWeightedCompressor:
    def __init__(self, grad, gamma, shape, use_ratio=0.1):
        self.grad = grad
        self.gamma = gamma
        self.use_ratio = use_ratio
        self.num_bits = int((use_ratio * 3 + 1) * shape)
        self.w = np.ones(shape=shape) / 2
    
    def __call__(self, x):
        x = np.copy(x)

        k = int(len(x) * self.use_ratio)
        d = len(x)

        GD = GradientDescentOnCube(self.w, self.grad, gamma_0=self.gamma, gamma=1.)

        GD.fit(x, max_iter=100)
        self.w = GD.w
        
        g_k = self.grad(x)
        indices = np.argsort(-self.w)[:k]
        
        weights = np.zeros_like(g_k)
        weights[indices] = self.w[indices]
        
        result, self.num_bits = greedy_compress(weights * g_k, indices)

        return result
    
class GradientDescentWeightedTopkCompressor:
    def __init__(self, grad, gamma, shape, use_ratio=0.1):
        self.grad = grad
        self.gamma = gamma
        self.use_ratio = use_ratio
        self.num_bits = int((use_ratio * 3 + 1) * shape)
        self.w = np.ones(shape=shape) / 2
    
    def __call__(self, x):
        x = np.copy(x)

        k = int(len(x) * self.use_ratio)
        d = len(x)

        GD = GradientDescentOnCube(self.w, self.grad, gamma_0=self.gamma, gamma=1.)

        GD.fit(x, max_iter=100)
        self.w = GD.w
        
        g_k = self.grad(x)
        indices = np.argsort(-self.w * np.abs(g_k))[:k]
        
        weights = np.zeros_like(g_k)
        weights[indices] = self.w[indices]
        
        result, self.num_bits = greedy_compress(weights * g_k, indices)

        return result

class GreedyCompressor:
    def __init__(self, grad, shape, use_ratio=0.1):
        self.grad = grad
        self.use_ratio = use_ratio
        self.num_bits = int((use_ratio * 3 + 1) * shape)
    
    def __call__(self, x):
        g_k = self.grad(x)

        k = int(len(x) * self.use_ratio)
        d = len(x)

        indices = np.argsort(-np.abs(g_k))[:k]
        result, self.num_bits = greedy_compress(g_k, indices)

        return result


class RandomCompressor:
    def __init__(self, grad, shape, use_ratio=0.1):
        self.grad = grad
        self.use_ratio = use_ratio
        self.num_bits = int((use_ratio * 3 + 1) * shape)
    
    def __call__(self, x):
        g_k = self.grad(x)

        indices = np.random.choice(len(g_k), int(len(g_k) * self.use_ratio))
        result, self.num_bits = greedy_compress(g_k, indices)

        return result


class BitNormGradientApproximator:
    def __init__(self, compressor, func_name="quadratic", args=None):
        self.func_name = func_name
        self.args = args
        self.compressor = compressor
        
        if self.func_name == "quadratic":
            self.A = self.args['A']
            self.b = self.args['b']
            self.c = self.args['c']
        elif self.func_name == "mushrooms":
            self.X = self.args['X']
            self.y = self.args['y']
            self.alpha = self.args['alpha']

        self.g_curr = None
        self.name = "Bit norm grad"
    
    def approx_gradient(self, x):
        if self.func_name == "quadratic":
            grad = utils.quadratic_grad(x, A=self.A, b=self.b)
        elif self.func_name == "mushrooms":
            grad = utils.logreg_grad(x, self.X, self.y, alpha=self.alpha)
        
        self.g_curr = self.compressor(x)
        
        return self.g_curr, self.compressor.num_bits