from src.perceptrons.multilayer_perceptron import MultilayerPerceptron
import numpy as np

class Autoencoder:
    def __init__(self, input, learning_rate:float,activation_function, activation_derivate, layer_structure, optimizer=None):
        self.input=np.array(input)
        self.encoder=MultilayerPerceptron(learning_rate,input,None,activation_function, activation_derivate, layer_structure,optimizer)
        self.decoder=MultilayerPerceptron(learning_rate,None,input,activation_function, activation_derivate, layer_structure[::-1],optimizer)

    def train(self,epochs,expected):
        for epoch in range(epochs):
            errors=0
            for i in range(len(self.input)):
                x=MultilayerPerceptron.get_input_with_bias(self.input[i])
                encoder_outputs,encoder_partials=self.encoder.feedfoward(x)

                z=MultilayerPerceptron.get_input_with_bias(encoder_outputs[-1])
                decoder_outputs, decoder_partials=self.decoder.feedfoward(z)

                self.decoder.backpropagate(decoder_partials,expected, decoder_outputs)
                self.encoder.backpropagate(encoder_partials,z, encoder_outputs)
                errors+=np.abs(self.input[i] - expected[i]).sum()
            
            print(f"epoch {epoch} error:{errors}")
            if errors<=1:
                print("Convergencia error")
                return

    