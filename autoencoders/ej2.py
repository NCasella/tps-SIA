import json
import pandas as pd 
import numpy as np
import sys
import matplotlib.pyplot as plt
from variational_autoencoder import VariationalAutoencoder
from src.perceptrons.multilayer_perceptron import MultilayerPerceptron
from src.perceptrons.sigmoid_functions import get_sigmoid_function_and_derivate
from variational_autoencoder import VariationalAutoencoder
from src.perceptrons.optimizers.optimizers import *
from fonts.fonts import *
from src.utils import *

if __name__=="__main__":
    fonts=[font_1,font_2,font_3]
    font_labels=[font1_labels,font2_labels,font3_labels]

    with open(sys.argv[1],"r") as f:
        config=json.load(f)
    layers_config=config["layers"]
    learning_rate=config["learning_rate"]
    function=config["activation_function"]
    f,df=get_sigmoid_function_and_derivate(1,function)
    epochs=config["epochs"]
    epsilon=config["epsilon"]
    optimizer_value=config["optimizer"]
    optimizer_alpha=config["optimizer_alpha"]
    optimizer_beta1=config["optimizer_beta_1"]
    optimizer_beta2=config["optimizer_beta_2"]
    optimizer_epsilon=config["optimizer_epsilon"]
    
    input=[to_bin_array(encoded_character).flatten() for encoded_character in fonts[2]]
    input_size = layers_config[0]
    layer_shapes = []
    current_size = input_size + 1
    for layer_size in layers_config[1:]:
        layer_shapes.append((current_size, layer_size))
        current_size = layer_size + 1
    encoder_layers=layer_shapes.copy()
    encoder_layers[-1]*=2
    decoder_layers=layer_shapes[::-1]
    enc_opt=get_optimizer(optimizer_value ,learning_rate,optimizer_alpha,optimizer_beta1,optimizer_beta2,optimizer_epsilon, encoder_layers)
    dec_opt=get_optimizer(optimizer_value ,learning_rate,optimizer_alpha,optimizer_beta1,optimizer_beta2,optimizer_epsilon, decoder_layers)



    encoder=MultilayerPerceptron(learning_rate,f,df,layers_config,enc_opt)
    decoder=MultilayerPerceptron(learning_rate,f,df,layers_config[::-1],dec_opt)


    vartiational_autoencoder: VariationalAutoencoder=VariationalAutoencoder(encoder,decoder)
    vartiational_autoencoder.train(input,epochs)