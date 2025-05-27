import numpy as np 

class Hopfield:
    def __init__(self,input_data , weights=None):
        self.input_data :np.ndarray =np.array(input_data)
        if weights is not None:
            self.weights :np.ndarray=np.array(weights)
        else:
            self.weights:np.ndarray= self.input_data.transpose()@self.input_data/self.input_data.shape[1]
            np.fill_diagonal(self.weights,0)
    
    def predic_output(self, input, epochs):
        current_state=np.array(input)
        state_history=[current_state]
        for epoch in range(epochs):
            current_state=np.sign(current_state@self.weights)
            state_history.append(current_state)
            if (len(state_history) > 3 and np.array_equal(state_history[-1], state_history[-2]) and np.array_equal(state_history[-2], state_history[-3])):
                return state_history
        return state_history
    
    def _get_energy_(self, state):
        sum=0
        for i in range(self.weights.shape[0]):
            for j in range(i+1, self.weights.shape[1]):
                sum+=self.weights[i][j]*state[i]*state[j]
        return sum