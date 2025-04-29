from simple_perceptron import SimplePerceptron

class LinearPerceptron(SimplePerceptron):
    
    def compute_activation(hμ):
        return hμ
    
    def calculate_error(self, x_set):
        raise NotImplementedError
