from simple_perceptron import LinearPerceptron


class NonLinearPerceptron(LinearPerceptron):
    def __init__(self,learning_rate:float,training_input:list,training_output: list,activation_function:callable,activation_function_derivate:callable):
       super.__init__(learning_rate,training_input,training_output)
       self.activation_function=activation_function
       self.activation_function_derivate=activation_function_derivate
       
    def compute_activation(self,hμ):
       return self.activation_function(hμ)
       
    def calculate_derivate(self,hμ):
        return self.activation_function_derivate(hμ)
    