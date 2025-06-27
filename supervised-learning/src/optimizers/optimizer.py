from abc import ABC,abstractmethod

class Optimizer(ABC):
    def __init__(self,learning_rate,alpha,beta1,beta2,epsilon,layers_structure):
        self.learning_rate=learning_rate
        self.alpha=alpha
        self.beta1=beta1
        self.beta2=beta2
        self.epsilon=epsilon
        self.layers_structure=layers_structure

    # returns the Delta W for each weight
    @abstractmethod
    def optimize(self, old_weight_adjustment, gradient, index):
        raise NotImplementedError
    