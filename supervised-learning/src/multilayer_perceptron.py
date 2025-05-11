from functools import partial
from typing import Any

import numpy as np
from src.non_linear_perceptron import NonLinearPerceptron
from src.optimizers.optimizer import Optimizer

class MultilayerPerceptron(NonLinearPerceptron):
   def __init__(self,learning_rate:float,training_input:list,training_output: list,activation_function:callable,activation_function_derivate:callable,layers_structure:list[int],optimizer:Optimizer = None):
      super().__init__(learning_rate,training_input,training_output,activation_function,activation_function_derivate)
      self.layers_structure=layers_structure
      self.optimizer = optimizer

      self.weights = []
      input_size = self.training_input.shape[1]
      self.weights.append(np.random.randn(input_size, layers_structure[0]) * np.sqrt(1.0 / input_size))
      for i in range(len(layers_structure) - 1):
         prev_size = layers_structure[i] + 1
         self.weights.append(
            np.random.randn(prev_size, layers_structure[i + 1]) * np.sqrt(1.0 / prev_size)
         )

      self.latest_adjustments = [np.zeros_like(w) for w in self.weights]

   def train_perceptron(self, epochs, epsilon):
      for epoch in range(epochs):
         total_error = 0
         for Xμ in range(len(self.training_input)):
            inputs = self.training_input[Xμ].reshape(1, -1)
            activations,partial_results =self._feedfoward(inputs)
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
   
   #feedfoward !!
   def _feedfoward(self, input)->tuple[list[Any],list[Any]]:
      outputs = [input]
      partial_results = []

      for i, weight_matrix in enumerate(self.weights):
         hi = np.dot(outputs[-1], weight_matrix)
         partial_results.append(hi)

         activated = self.activation_function(hi)
         if i != len(self.weights) - 1:
            activated = np.hstack([
               activated,
               np.ones((activated.shape[0], 1))
            ])

         outputs.append(activated)

      return outputs, partial_results

   def _backpropagate(self, h_is, expected, outputs):
      deltas = []

      output = outputs[-1]
      h_i = h_is[-1]
      delta = (expected - output) * self.calculate_derivate(h_i)
      deltas.insert(0, delta)

      for l in range(len(self.weights) - 2, -1, -1):
         h_i = h_is[l]
         weights = self.weights[l + 1][:-1, :]
         delta = np.dot(deltas[0], weights.T) * self.calculate_derivate(h_i)
         deltas.insert(0, delta)

      for l in range(len(self.weights)):
         layer_output = outputs[l]
         delta = deltas[l]
         raw_gradient = np.outer(layer_output, delta)
         adjustment = self.optimizer.optimize(
            self.latest_adjustments[l],
            raw_gradient,
            l
         )
         self.weights[l] += adjustment
         self.latest_adjustments[l] = adjustment

   def _get_input_with_bias(self, training_input):
      if isinstance(training_input, list):
         training_input = np.array(training_input)
      if training_input.ndim == 1:
         training_input = training_input.reshape(1, -1)
      return np.hstack([training_input, np.ones((training_input.shape[0], 1))])

   def predict_output(self, input):
      inputs = self._get_input_with_bias(input)
      return self._feedfoward(inputs)[0][-1]

   def calculate_error(self, expected, output)->float:
      return 0.5*np.square(output-expected).sum()