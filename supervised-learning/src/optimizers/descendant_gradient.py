import numpy as np 
from src.optimizers.optimizer import Optimizer

class DescendantGradient(Optimizer):
    def __init__(self, learning_rate, alpha, beta1, beta2, epsilon):
        super().__init__(learning_rate, alpha, beta1, beta2, epsilon)
        
    def optimize(self, weights, gradient_w):
        raise NotImplementedError