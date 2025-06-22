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
    encode_layers_config=config["encoder_layers"]
    decode_layers_config=config["decoder_layers"]
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
    

    enc_input_size,dec_input_size = encode_layers_config[0],decode_layers_config[0]
    encode_layer_shapes = []
    decode_layer_shapes=[]
    enc_current_size = enc_input_size + 1
    dec_current_size=dec_input_size +1

    for enc_layer_size,dec_layer_size in zip(encode_layers_config[1:],decode_layers_config[1:]):
        encode_layer_shapes.append((enc_current_size, enc_layer_size))
        decode_layer_shapes.append((dec_current_size, dec_layer_size))
        enc_current_size=enc_layer_size + 1
        dec_current_size=dec_layer_size + 1
    
    enc_opt=get_optimizer(optimizer_value ,learning_rate,optimizer_alpha,optimizer_beta1,optimizer_beta2,optimizer_epsilon, encode_layer_shapes)
    dec_opt=get_optimizer(optimizer_value ,learning_rate,optimizer_alpha,optimizer_beta1,optimizer_beta2,optimizer_epsilon, decode_layer_shapes)



    encoder=MultilayerPerceptron(learning_rate,f,df,encode_layers_config,enc_opt)
    decoder=MultilayerPerceptron(learning_rate,f,df,decode_layers_config,dec_opt)

    input=[to_bin_array(encoded_character).flatten() for encoded_character in fonts[2]]

    vartiational_autoencoder: VariationalAutoencoder=VariationalAutoencoder(encoder,decoder)
    vartiational_autoencoder.train(input,epochs)