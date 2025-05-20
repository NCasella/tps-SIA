import numpy as np 

class Oja:
    def __init__(self, input_data:list, initial_learning_rate:float=1e-3, weights=None ):
        self.input_data=np.array(input_data)
        self.learning_rate=initial_learning_rate
        self.weights=weights if weights is not None else np.random.uniform(size=self.input_data.shape[1])
    
    def train_network(self, epochs):
        for epoch in epochs:
            for µ in range(len(self.input_data)):
                output=self.input_data[µ]@self.weights
                for i in range(len(self.weights)):
                    self.weights[i]+=self.learning_rate*output*(self.input_data[µ][i] -output*self.weights[i])
            self.learning_rate=self.learning_rate/epoch if self.learning_rate !=1e-3 else self.learning_rate 
            