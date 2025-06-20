import numpy as np 
from src.perceptrons.multilayer_perceptron import MultilayerPerceptron


#DEPRECADO (maybe)
class Autoencoder():
    def __init__(self,learning_rate:float,activation_function:callable,activation_function_derivate:callable,enc_layers_structure:list[int],dec_layers_structure,enc_optimizer = None,dec_optimizer=None):
        self.encoder:MultilayerPerceptron=MultilayerPerceptron(learning_rate,activation_function,activation_function_derivate,enc_layers_structure,enc_optimizer)
        self.decoder:MultilayerPerceptron=MultilayerPerceptron(learning_rate,activation_function,activation_function_derivate,dec_layers_structure,dec_optimizer)


    def train(self,training_input,training_output,epsilon,epochs):
        training_input=MultilayerPerceptron.get_input_with_bias(training_input)
        training_output=np.array(training_output)
        for epoch in range(epochs):
            errors=0
            for i in len(training_input):
                enc_out, enc_partials=self.encoder.feedfoward(training_input[i])

                z_input=MultilayerPerceptron.get_input_with_bias(enc_out[-1])
                dec_out,dec_partials=self.decoder.feedfoward(z_input)

                self.decoder.backpropagate(dec_partials,training_output[i],dec_out)
                self.encoder.backpropagate(enc_partials,enc_out[-1],enc_out)
                errors+=self.calculate_error(training_output[-1],dec_out[-1])
            if errors<=epsilon:
                print(f"Convergencia en {epoch}")
                return 
            print(f"epoch {epoch} error:{errors}")
    
    def calculate_error(self, expected, output)->float:
        return np.sum((output>0.5).astype(int)!=expected.astype(int))



