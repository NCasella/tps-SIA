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

def generate_new_letters(autoencoder, start_latent, end_latent, steps=10):
    """
    Interpolates between two latent coordinates and generates new letters.
    """
    interpolated = [
        start_latent + (end_latent - start_latent) * t
        for t in np.linspace(0, 1, steps)
    ]
    for idx, latent in enumerate(interpolated):
        output, _ = autoencoder.decode(latent)
        save_letter_heatmap(output[-1], f"out/interpolated_{idx}.png")
    

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
    font_number=int(config["font"])-1

    input=[to_bin_array(encoded_character).flatten() for encoded_character in fonts[font_number]]
    input_size = layers_config[0]
    layer_shapes = []
    current_size = input_size + 1
    for layer_size in layers_config[1:]:
        layer_shapes.append((current_size, layer_size))
        current_size = layer_size + 1
    opt=get_optimizer(optimizer_value ,learning_rate,optimizer_alpha,optimizer_beta1,optimizer_beta2,optimizer_epsilon,layer_shapes)
    input=np.array(input)

    errors_after_3k = [0 for _ in range(10)]

    seeds = [24680, 12345, 13579, 67890, 77777, 54321, 98765, 19283, 37465, 21378]

    for i in range(10):
        autoencoder:MultilayerPerceptron=MultilayerPerceptron(learning_rate,f,df,layers_config,opt,seed=seeds[i])
        autoencoder.train_perceptron(input,input,epochs,epsilon=epsilon)

        total_error=0
        for i,char in enumerate(input):
            output,latent_space_coord =autoencoder.predict_output(char) 
            total_error += autoencoder.calculate_error(char, output)
            # coords.append(latent_space_coord)
            # save_letter_heatmap(output,f"out/{font_labels[font_number][i]}.png")
        # plot_latent_space(coords,font_labels[font_number])
        if total_error < 9:
            errors_after_3k[total_error] += 1
        else:
            errors_after_3k[9] += 1
        print(f"Error total: {total_error}")
    print(f"Capas: {layers_config}")
    print("Errores despues de 3k iteraciones:")
    for i, count in enumerate(errors_after_3k):
        print(f"Error {i}: {count} veces")
    
        


