import numpy as np
from src.non_linear_perceptron import NonLinearPerceptron

class MultilayerPerceptron(NonLinearPerceptron):
   def __init__(self,learning_rate:float,training_input:list,training_output: list,activation_function:callable,activation_function_derivate:callable,layers_structure:list[int]):
      super().__init__(learning_rate,training_input,training_output,activation_function,activation_function_derivate)
      self.layers_structure=layers_structure
      self.weights=[]
      self.weights.append(np.random.randn(self.training_input.shape[1],layers_structure[0]))
      for i in range(len(layers_structure)-1):
         self.weights.append(np.random.randn((layers_structure[i],layers_structure[i+1])) ) 
   
   def train_perceptron(self, epochs, epsilon):   
      raise NotImplementedError
   def _feedfoward_propagation(self,input):
      raise NotImplementedError
   
   def _backtrack(self,expected,input):
      raise NotImplementedError