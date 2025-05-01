from src.simple_perceptron import SimplePerceptron

class LinearPerceptron(SimplePerceptron):
    
    def compute_activation(self,hμ):
        return hμ
    
    def calculate_error(self,expected, output):
        return 0.5*((expected-output))**2
