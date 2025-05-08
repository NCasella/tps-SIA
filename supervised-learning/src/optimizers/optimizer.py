import numpy as np 
from abc import ABC,abstractmethod

class Optimizer(ABC):
    def __init__(self,learning_rate,alpha,beta1,beta2,epsilon):
        self.learning_rate=learning_rate
        self.alpha=alpha
        self.beta1=beta1
        self.beta2=beta2
        self.epsilon=epsilon
        self.weights=None
    
    @abstractmethod
    def optimize(self,weights:np.matrix,gradient_w:np.matrix)->np.matrix:
        raise NotImplementedError
    