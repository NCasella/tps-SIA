import numpy as np 
from src.optimizers.optimizer import Optimizer

class GradientDescent(Optimizer):
    def __init__(self, learning_rate, alpha, beta1, beta2, epsilon, all_layers):
        super().__init__(learning_rate, alpha, beta1, beta2, epsilon, all_layers)
        
    def optimize(self, old_weight_adjustment:float, gradient_w:float, index: tuple[int,int,int]) -> float:
        return self.learning_rate * gradient_w