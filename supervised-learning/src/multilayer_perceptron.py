from typing import Any

import numpy as np
from src.non_linear_perceptron import NonLinearPerceptron
from src.optimizers.optimizer import Optimizer

class MultilayerPerceptron(NonLinearPerceptron):
   def __init__(self,learning_rate:float,training_input:list,training_output: list,activation_function:callable,activation_function_derivate:callable,layers_structure:list[int],optimizer:Optimizer = None):
      super().__init__(learning_rate,training_input,training_output,activation_function,activation_function_derivate)
      self.layers_structure=layers_structure

      all_layers = np.concatenate(([len(training_input[0])], layers_structure))

      self.weights = [np.random.randn(all_layers[i], all_layers[i + 1]) for i in range(len(all_layers) - 1)]
      self.latest_adjustments = [np.zeros((all_layers[i], all_layers[i + 1])) for i in range(len(all_layers) - 1)]
      self.optimizer=optimizer

   def train_perceptron(self, epochs, epsilon):
      for epoch in range(epochs):
         total_error = 0
         for Xμ in range(len(self.training_input)):
            activations,partial_results =self.predict_output(self.training_input[Xμ])
            # print(f"Activations: {activations}")
            # print(f"Partial results: {partial_resuls}")
            # print(f"Expected: {self.training_output[Xμ]}")
            self._backpropagate(partial_results, self.training_output[Xμ], activations)
            total_error += self.calculate_error(self.training_output[Xμ], activations[-1])
         if total_error<epsilon:
            print(f"Convergencia en epoch {epoch}")
            return
         print(f"epoch: {epoch} y Error:{total_error}")
         permutation=np.random.permutation(len(self.training_input))
         self.training_input=self.training_input[permutation]
         self.training_output=self.training_output[permutation]
      print("No convergencia :(")
   
   #feedfoward (?
   def predict_output(self, input)-> tuple[list[Any], list[Any]]:
      h_o=np.vectorize(self.activation_function)
      outputs=[input]
      partial_results=[]
      for weight_matrix in self.weights:
         hi = np.matmul(outputs[-1],weight_matrix)
         partial_results.append(hi)
         outputs.append(h_o(hi))
      return outputs, partial_results
   
   def _backpropagate(self, h_is, expected, outputs):
      output = outputs[-1]
      h_i = h_is[-1]
      Vj = outputs[-2]
      weight_matrix = self.weights[-1]
      old_adjustment_matrix = self.latest_adjustments[-1]
      delta_matrix = [[] for _ in self.layers_structure]
      if len(weight_matrix[0]) != len(h_i):
         raise ValueError("Weight matrix must have as many rows as the number of h_i calculated")
      for i in range(len(output)):
         delta = (expected[i] - output[i]) * self.calculate_derivate(h_i[i])

         delta_matrix[len(self.layers_structure) - 1].append(delta)
         for j in range(len(weight_matrix[i])):
            gradient_w = delta * Vj[j]
            old_adjustment = old_adjustment_matrix[i][j]
            delta_w = self.optimizer.optimize(old_adjustment, gradient_w, (-1, i, j))
            old_adjustment_matrix[i][j] = delta_w
            weight_matrix[i][j] += delta_w
      
      V_sizes = self.layers_structure[:-1][::-1]
      for idx in range(len(V_sizes)-1):
         m = len(V_sizes) - idx - 1
         Vj = outputs[m]
         h_i = h_is[m]
         weight_matrix = self.weights[m]
         old_adjustment_matrix = self.latest_adjustments[m]
         Vk = outputs[m - 1]
         for j in range(len(Vj)):
            delta = self.calculate_derivate(h_i[j]) * np.sum(delta_matrix[m + 1] * self.weights[m + 1][j])
            # delta_matrix[m][j] = delta
            delta_matrix[m].append(delta)
            for k in range(len(weight_matrix[j])):
               gradient_w = delta * Vk[k]
               old_adjustment = old_adjustment_matrix[j][k]
               delta_w = self.optimizer.optimize(old_adjustment, gradient_w, (m, j, k))
               old_adjustment_matrix[j][k] = delta_w
               weight_matrix[j][k] += delta_w


   def calculate_error(self, expected, output)->float:
      return 0.5*np.square(output-expected).sum()