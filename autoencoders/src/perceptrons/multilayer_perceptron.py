from typing import Any
import numpy as np
from src.perceptrons.optimizers.optimizer import Optimizer

class MultilayerPerceptron():
   def __init__(self,learning_rate:float,activation_function:callable,activation_function_derivate:callable,layers_structure:list[int],optimizer:Optimizer = None):
      self.activation_function=activation_function
      self.learning_rate=learning_rate
      self.activation_function_derivate=activation_function_derivate
      self.layers_structure=layers_structure
      self.optimizer = optimizer
      self.weights = []
      input_size = layers_structure[0]+1
      self.weights.append(np.random.randn(input_size, layers_structure[1]) * np.sqrt(1.0 / input_size))
      for i in range(1,len(layers_structure) - 1):
         prev_size = layers_structure[i] + 1
         self.weights.append(
            np.random.randn(prev_size, layers_structure[i + 1]) * np.sqrt(1.0 / prev_size)
         )

      self.latest_adjustments = [np.zeros_like(w) for w in self.weights]

   def train_perceptron(self,input,output, epochs, epsilon):
      training_input=np.array(input)
      training_input=MultilayerPerceptron.get_input_with_bias(training_input)
      training_output=np.array(output)
      for epoch in range(epochs):
         total_error = 0
         for Xμ in range(len(training_input)):
            inputs = training_input[Xμ].reshape(1, -1)
            activations,partial_results =self.feedfoward(inputs)
            self.backpropagate(partial_results, training_output[Xμ], activations)
            total_error += self.calculate_error(training_output[Xμ], activations[-1])
         if total_error<epsilon:
            print(f"Convergencia en epoch {epoch}")
            return
         print(f"epoch: {epoch} y Error:{total_error}")
         permutation=np.random.permutation(len(training_input))
         training_input=training_input[permutation]
         training_output=training_output[permutation]
      print("No convergencia :(")
   
   #feedfoward !!
   def feedfoward(self, input)->tuple[list[Any],list[Any]]:
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

   def backpropagate(self, h_is, expected, outputs):
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

   @classmethod
   def get_input_with_bias(self, training_input):
      if isinstance(training_input, list):
         training_input = np.array(training_input)
      if training_input.ndim == 1:
         training_input = training_input.reshape(1, -1)
      return np.hstack([training_input, np.ones((training_input.shape[0], 1))])

   def predict_output(self, input):
      inputs = MultilayerPerceptron.get_input_with_bias(input)
      feedfoward=self.feedfoward(inputs)[0]
      return feedfoward[-1],feedfoward[len(self.layers_structure)//2][0][:-1]

   def decode(self,input):
      input=MultilayerPerceptron.get_input_with_bias(input) 
      outputs = [input]
      partial_results = []
      
      for i, weight_matrix in enumerate(self.weights[len(self.weights)//2:]):
         hi = np.dot(outputs[-1], weight_matrix)
         partial_results.append(hi)

         activated = self.activation_function(hi)
         if i != len(self.weights[len(self.weights)//2:]) - 1:
            activated = np.hstack([
               activated,
               np.ones((activated.shape[0], 1))
            ])

         outputs.append(activated)

      return outputs, partial_results



   def calculate_error(self, expected, output)->float:
      return np.sum((output>0.5).astype(int)!=expected.astype(int))
   
   def compute_activation(self,hμ):
      return self.activation_function(hμ)

   def calculate_derivate(self,hμ):
      return self.activation_function_derivate(hμ)
