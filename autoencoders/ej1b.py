import json
import pandas as pd 
import numpy as np
import sys
import matplotlib.pyplot as plt
from src.perceptrons.multilayer_perceptron import MultilayerPerceptron
from src.perceptrons.sigmoid_functions import get_sigmoid_function_and_derivate
from src.perceptrons.optimizers.optimizers import *
from fonts.fonts import *
from src.utils import *
from src.noise_functions import get_noise_functions


if __name__=="__main__":
    fonts=[font_1,font_2,font_3]


    with open(sys.argv[1],"r") as f:
        config=json.load(f)
    layers_config=config["layers"]
    epsilon=config["epsilon"]
    std_deviation=config["standard_deviation"]
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
    noise_function=config["noise_function"]


    input=[to_bin_array(encoded_character).flatten() for encoded_character in fonts[font_number]]
    input_size = layers_config[0]
    layer_shapes = []
    current_size = input_size + 1
    for layer_size in layers_config[1:]:
        layer_shapes.append((current_size, layer_size))
        current_size = layer_size + 1
    opt=get_optimizer(optimizer_value ,learning_rate,optimizer_alpha,optimizer_beta1,optimizer_beta2,optimizer_epsilon,layer_shapes)
    input=np.array(input)
    noise_func=get_noise_functions(noise_function, std_deviation)

    noisy_input=noise_func(input)
    autoencoder:MultilayerPerceptron=MultilayerPerceptron(learning_rate,f, df, layers_config,opt)
    autoencoder.train_perceptron(noisy_input,input,epochs,epsilon=epsilon)

    for i,char in enumerate(noise_func(input)):
        output,_ =autoencoder.predict_output(char) 
        save_letter_heatmap(char,f"out/noisy{font2_labels[i]}.png",binarize=False)
        save_letter_heatmap(output,f"out/{font2_labels[i]}.png")
