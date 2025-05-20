import numpy as np
from typing import Callable,Optional
import random
class Kohonen:
    def __init__(self, grid_size:int, input_data:list,initial_learning_rate:float,similarity_metric:Callable,radius:Optional[float]=None, weights=None ):
        self.grid_size: int =grid_size
        self.learning_rate: float=initial_learning_rate
        self.radius:float=radius if radius is not None else grid_size
        self.similarity_metric: Callable =similarity_metric
        self.input_data: np.ndarray =np.array(input_data)
        self.weights: np.ndarray = weights if weights is not None else np.random.uniform(-0.5,0.5,(grid_size**2,self.input_data.shape[1]) )
        
    def train_network(self, iterations: int):
        for i in range(iterations):
            for X in self.input_data:
                winner_neuron_index=self.similarity_metric(X,self.weights)
                nearby_neurons=self._get_nearby_neurons(winner_neuron_index)
                self._update_weights(X,nearby_neurons,self.learning_rate/(i+1))
                
                
    def _update_weights(self,input_data,neighbours,current_learning_rate):
        for n in neighbours:
            self.weights[n]+=current_learning_rate*(input_data-self.weights[n])
    
    def _get_nearby_neurons(self,winner_neuron_index):
        nearby_neurons_indexes=[]
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (i-winner_neuron_index//self.grid_size)**2 +(j-winner_neuron_index%self.grid_size)**2 < self.radius**2:
                    nearby_neurons_indexes.append(i*self.grid_size+j)
        
        return nearby_neurons_indexes