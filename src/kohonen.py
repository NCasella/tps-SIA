import numpy as np
from typing import Callable,Optional

class Kohonen:
    def __init__(self, grid_size:int, input_data:list,similarity_metric:Callable,radius:Optional[float]=None,constant_radius:bool=False ,weights=None ):
        self.grid_size: int =grid_size
        self.radius:float=radius if radius is not None else grid_size
        self.similarity_metric: Callable =similarity_metric
        self.input_data: np.ndarray =np.array(input_data)
        self.constant_radius=constant_radius
        """Matriz de pesos NxD 
            N: cantidad de neuronas (grid_size^2)
            D: dimension de las entradas"""
        self.weights: np.ndarray = weights if weights is not None else np.random.uniform(-0.5,0.5,(grid_size**2,self.input_data.shape[1]) )
        
    def train_network(self, iterations: int):
        weight_history=[self.weights]
        for i in range(iterations):
            X=self.input_data[np.random.choice(len(self.input_data))]
            winner_neuron_index=self.similarity_metric(X,self.weights)
            nearby_neurons=self._get_nearby_neurons(winner_neuron_index)
            self._update_weights(X,nearby_neurons,1/(i+1))
            weight_history.append(self.weights)
        return weight_history
    
    def map_input(self,input=None):
        data =self.input_data if input is None else np.array(input)
        coord_results=[]
        for x in data:
            best_neuron_index=self.similarity_metric(x, self.weights)
            coord_results.append((best_neuron_index//self.grid_size, best_neuron_index % self.grid_size))
        return coord_results

    def _update_weights(self,input_data,neighbours_indexes,current_learning_rate):
        for n in neighbours_indexes:
            self.weights[n]+=current_learning_rate*(input_data-self.weights[n])
    
    def _get_nearby_neurons(self,winner_neuron_index):
        nearby_neurons_indexes=[]
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (i-winner_neuron_index//self.grid_size)**2 +(j-winner_neuron_index%self.grid_size)**2 < self.radius**2:
                    nearby_neurons_indexes.append(i*self.grid_size+j)
        if not self.constant_radius:
            self.radius=max(1,self.radius*0.90)
        
        return nearby_neurons_indexes