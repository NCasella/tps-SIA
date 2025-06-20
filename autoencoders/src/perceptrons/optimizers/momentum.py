import numpy as np 
from src.perceptrons.optimizers.optimizer import Optimizer

class Momentum(Optimizer):
    def optimize(self, old_adjustment, raw_gradient, layer_index):
        return self.alpha * old_adjustment + self.learning_rate * raw_gradient