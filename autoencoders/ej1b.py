import json
import pandas as pd 
import numpy as np
import sys
import matplotlib.pyplot as plt
from src.autoencoder import Autoencoder
from src.perceptrons.multilayer_perceptron import MultilayerPerceptron
from src.perceptrons.sigmoid_functions import get_sigmoid_function_and_derivate
from src.perceptrons.optimizers.optimizers import *
from fonts.fonts import *
from src.utils import *



if __name__=="__main__":
    fonts=[font_1,font_2,font_3]


    with open(sys.argv[1],"r") as f:
        config=json.load(f)
    layers_config=config["layers"]
    learning_rate=config["learning_rate"]
    function=config["activation_function"]
    f,df=get_sigmoid_function_and_derivate(1,function)
    epochs=config["epochs"]
    optimizer_value=config["optimizer"]
    optimizer_alpha=config["optimizer_alpha"]
    optimizer_beta1=config["optimizer_beta_1"]
    optimizer_beta2=config["optimizer_beta_2"]
    optimizer_epsilon=config["optimizer_epsilon"]
    font_number=int(config["font"])-1
    
    input=[to_bin_array(encoded_character).flatten() for encoded_character in fonts[font_number]]
    input_size = len(input[0])
    layer_shapes = []
    current_size = input_size + 1
    for layer_size in layers_config:
        layer_shapes.append((current_size, layer_size))
        current_size = layer_size + 1
    opt=get_optimizer(optimizer_value ,learning_rate,optimizer_alpha,optimizer_beta1,optimizer_beta2,optimizer_epsilon,layer_shapes)
    input=np.array(input)


    autoencoder:MultilayerPerceptron=MultilayerPerceptron(learning_rate,input,input,f,df,layers_config,opt)