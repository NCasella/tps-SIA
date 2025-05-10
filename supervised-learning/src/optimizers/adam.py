import numpy as np
from src.optimizers.optimizer import Optimizer

class Adam(Optimizer):
    def __init__(self, learning_rate, alpha, beta1, beta2, epsilon, all_layers):
        super().__init__(learning_rate, alpha, beta1, beta2, epsilon)
        self.m_vector = [np.zeros((all_layers[i], all_layers[i + 1])) for i in range(len(all_layers) - 1)]
        self.v_vector = [np.zeros((all_layers[i], all_layers[i + 1])) for i in range(len(all_layers) - 1)]
        self.theta_vector = [np.zeros((all_layers[i], all_layers[i + 1])) for i in range(len(all_layers) - 1)]
        self.current_iteration = 1

    def optimize(self, old_weight_adjustment: float, gradient_w: float, index: tuple[int,int,int]) -> float:
        i,j,k = index

        m_t = self.beta1 * self.m_vector[i][j][k] + (1 - self.beta2) * gradient_w
        v_t = self.beta2 * self.v_vector[i][j][k] + (1 - self.beta2) * gradient_w * gradient_w
        m_hat = m_t / (1 - self.beta1 ** self.current_iteration)
        v_hat = v_t / (1 - self.beta2 ** self.current_iteration)
        self.m_vector[i][j][k] = m_t
        self.v_vector[i][j][k] = v_t
        return self.theta_vector[i][j][k] - self.alpha / (np.sqrt(v_hat) + self.epsilon) * m_hat

