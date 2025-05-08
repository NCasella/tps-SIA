import numpy as np
from src.non_linear_perceptron import NonLinearPerceptron
from src.optimizers.optimizer import Optimizer

class MultilayerPerceptron(NonLinearPerceptron):
   def __init__(self,learning_rate:float,training_input:list,training_output: list,activation_function:callable,activation_function_derivate:callable,layers_structure:list[int],optimizer:Optimizer):
      super().__init__(learning_rate,training_input,training_output,activation_function,activation_function_derivate)
      self.layers_structure=layers_structure
      self.weights=[]
      self.optimizer=optimizer
      self.weights.append(np.random.randn(self.training_input.shape[1],layers_structure[0]))
      for i in range(len(layers_structure)-1):
         self.weights.append(np.random.randn((layers_structure[i],layers_structure[i+1])) ) 
   
   def train_perceptron(self, epochs, epsilon):
      for epoch in range(epochs):
         total_error = 0
         for Xμ in self.training_input:
            hμ,activations =self.predict_output(Xμ)
            #TODO
         if total_error<epsilon:
            print(f"Convergencia en epoch {epoch}")
            return
         print(f"epoch: {epoch} y Error:{total_error}")
         permutation=np.random.permutation(len(self.training_input))
         self.training_input=self.training_input[permutation]
         self.training_output=self.training_output[permutation]
      print("No convergencia :(")
   
   #feedfoward (?
   def predict_output(self, input)->tuple[np.array, list[np.array]]:
      h_o=np.vectorize(self.activation_function)
      outputs=[input]
      for weight_matrix in self.weights:
         outputs.append(h_o(np.matmul(outputs[-1],weight_matrix)))
      return outputs[-1],outputs
   
   def _backtrack(self,output,expected)->list[np.matrix]:
      raise NotImplementedError
   
   def calculate_error(self, expected, output)->float:
      return 0.5*np.square(output-expected).sum()