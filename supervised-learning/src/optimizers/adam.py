import numpy as np
from src.optimizers.optimizer import Optimizer

class Adam(Optimizer):
    def __init__(self, learning_rate, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=10e-8):
        super().__init__(learning_rate, alpha, beta1, beta2, epsilon)
    
    def optimize(self, weights, gradient_w):
        raise NotImplementedError