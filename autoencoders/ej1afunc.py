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

def plot_latent_space(X, labels=None, title="Latent Space", figsize=(7, 5)):

    latent_points=np.array(X)
    plt.figure(figsize=figsize)
    plt.scatter(latent_points[:, 0], latent_points[:, 1], c='blue', edgecolors='k')

    if labels is not None:
        for i, (x, y) in enumerate(latent_points):
            plt.text(x, y+0.08, str(labels[i]), fontsize=9, ha='center', va='center', color='black')

    plt.title(title)
    plt.xlabel("z₁")
    plt.ylabel("z₂")
    plt.grid(True)
    plt.savefig("latent_space.png")


if __name__=="__main__":
    fonts=[font_1,font_2,font_3]
    font_labels=[font1_labels,font2_labels,font3_labels]

    with open(sys.argv[1],"r") as f:
        config=json.load(f)
    layers_config=config["layers"]
    learning_rate=config["learning_rate"]
    function=config["activation_function"]
    # f,df=get_sigmoid_function_and_derivate(1,function)
    epochs=config["epochs"]
    epsilon=config["epsilon"]
    optimizer_value=config["optimizer"]
    optimizer_alpha=config["optimizer_alpha"]
    optimizer_beta1=config["optimizer_beta_1"]
    optimizer_beta2=config["optimizer_beta_2"]
    optimizer_epsilon=config["optimizer_epsilon"]
    font_number=int(config["font"])-1

    errors_after_5k = [0 for _ in range(10)]

    activation_functions = ["logistic", "tanh", "relu", "softplus"]
    f, df = get_sigmoid_function_and_derivate(1,"tanh")

    layer_configs1 = [
        [35,2,35],
        [35, 10, 2, 10, 35],
        [35, 18, 2, 18, 35],
        [35, 25, 2, 25, 35],
        [35, 30, 2, 30, 35],
        [35, 50, 2, 50, 35],
        [35, 80, 2, 80, 35]
    ]
    layer_config_strings1 = [
        "35-2-35", 
        "35-10-2-10-35",
        "35-18-2-18-35",
        "35-25-2-25-35",
        "35-30-2-30-35",
        "35-50-2-50-35",
        "35-80-2-80-35"
    ]

    layer_configs = [
        [35, 20, 10, 2, 10, 20, 35],
        [35, 30, 20, 2, 20, 30, 35],
        [35, 32, 28, 2, 28, 32, 35],
        [35, 50, 20, 2, 20, 50, 35],
        [35, 80, 50, 2, 50, 80, 35],
    ]
    layer_config_strings = [
        "35-20-10-2-10-20-35",
        "35-30-20-2-20-30-35",
        "35-32-28-2-28-32-35",
        "35-50-20-2-20-50-35",
        "35-80-50-2-50-80-35",
    ]

    layer_configs3 = [
        [35, 30, 20, 10, 2, 10, 20, 30, 35],
        [35, 32, 28, 20, 2, 20, 28, 32, 35],
        [35, 50, 30, 20, 2, 20, 30, 50, 35],
        [35, 40, 30, 20, 2, 20, 30, 40, 35],
        [35, 80, 50, 30, 2, 30, 50, 80, 35],
    ]
    layer_config_strings3 = [
        "35-30-20-10-2-10-20-30-35",
        "35-32-28-20-2-20-28-32-35",
        "35-50-30-20-2-20-30-50-35",
        "35-40-30-20-2-20-30-40-35",
        "35-80-50-30-2-30-50-80-35",
    ]

    optimizers = ["adam", "sgd", "momentum"]

    for conf in range(5):
        layers_config = layer_configs[conf]
        # optimizer_value = optimizers[conf]
        input=[to_bin_array(encoded_character).flatten() for encoded_character in fonts[font_number]]
        input_size = layers_config[0]
        layer_shapes = []
        current_size = input_size + 1
        for layer_size in layers_config[1:]:
            layer_shapes.append((current_size, layer_size))
            current_size = layer_size + 1
        opt=get_optimizer(optimizer_value ,learning_rate,optimizer_alpha,optimizer_beta1,optimizer_beta2,optimizer_epsilon,layer_shapes)
        input=np.array(input)

        autoencoder:MultilayerPerceptron=MultilayerPerceptron(learning_rate,f,df,layers_config,opt)
        errors = autoencoder.train_perceptron(input,input,epochs,epsilon=epsilon)
        coords=[]

        plt.plot(errors, label=f"{layer_config_strings[conf]}")
        total_error = 0
        for i,char in enumerate(input):
            output,latent_space_coord =autoencoder.predict_output(char) 
            total_error += autoencoder.calculate_error(char, output)

    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.title("Error vs Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig("7layers_comparison.png")
    plt.show()


