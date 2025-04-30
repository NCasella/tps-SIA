import numpy as np
import random
import math
class SimplePerceptron():
    def __init__(self,learning_rate:float=1e-4,training_input:list=None,training_output: list=None): 
        self.learning_rate=learning_rate
        self.training_input=self._get_input_with_bias(training_input)
        self.weights=[ random.uniform(-0.5,0.5) for w in range(self.training_input.shape[1]) ]
        self.training_output=training_output
    
    def _get_input_with_bias(self, training_input):
        training_input = np.array(training_input)
        if training_input.ndim == 1:  
            training_input = training_input.reshape(1, -1)
        bias = np.ones((training_input.shape[0], 1), dtype=int)  
        return np.hstack([bias, training_input])  
        
    def calculate_error(self,expected,output):
        return abs(expected-output)
    
    def compute_activation(self,hμ):
        return 1 if hμ>=0 else -1
    
    def calculate_derivate(self,hµ):
        return 1
    
    def predict_output(self, input):
        x_with_bias = self._get_input_with_bias(input)
        return self.compute_activation(x_with_bias @ np.array(self.weights))

    
    def train_perceptron(self,epochs:int,epsilon:float):
        convergence:bool=True
        for epoch in range(epochs):
            for µ in range(len(self.training_input)): #cada x^µ
                hµ:float=self.training_input[μ]@self.weights
                o_h=self.compute_activation(hμ)
                for i,w_i in enumerate(self.weights):
                    self.weights[i]=w_i+self.learning_rate*(self.training_output[µ]-o_h)*self.calculate_derivate(hμ)*self.training_input[μ][i]
                error=self.calculate_error(expected=self.training_output[μ],output=o_h)                                     #^^^^^^^^^^^^^^^^^^ (X^µ)_i
                convergence=error<epsilon #TODO: ver que hacer con el break
        if convergence:
            print("se llego a convergencia ")
        else:
            print("no se llego a convergencia :(")