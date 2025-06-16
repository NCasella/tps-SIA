import numpy as np
from src.optimizers.optimizer import Optimizer

class Adam(Optimizer):
    def __init__(self, learning_rate, alpha, beta1, beta2, epsilon, all_layers):
        super().__init__(learning_rate, alpha, beta1, beta2, epsilon, all_layers)

        self.m = [np.zeros(shape) for shape in all_layers]
        self.v = [np.zeros(shape) for shape in all_layers]
        self.t = 0

    def optimize(self, old_adjustment, raw_gradient, layer_index):
        self.t += 1
        layer = layer_index

        self.m[layer] = self.beta1 * self.m[layer] + (1 - self.beta1) * raw_gradient
        self.v[layer] = self.beta2 * self.v[layer] + (1 - self.beta2) * (raw_gradient ** 2)

        m_hat = self.m[layer] / (1 - self.beta1 ** self.t)
        v_hat = self.v[layer] / (1 - self.beta2 ** self.t)

        return (self.learning_rate * m_hat) / (np.sqrt(v_hat) + self.epsilon)

