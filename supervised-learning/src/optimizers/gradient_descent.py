import numpy as np 
from src.optimizers.optimizer import Optimizer

class GradientDescent(Optimizer):
        
    def optimize(self, old_adjustment, raw_gradient, layer_index):
        return self.learning_rate * raw_gradient