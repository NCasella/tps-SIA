import json
import sys
from src.kohonen import Kohonen
from src.similarity_metrics import get_metric_function
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 

def plot_data_mapping(som: Kohonen, labels=None):
    grid_size = som.grid_size
    mapped = som.map_input()
    
    plt.figure(figsize=(6, 6))
    for idx, neuron_index in enumerate(mapped):
        x, y = neuron_index
        if labels is not None:
            plt.text(y, x, str(labels[idx]), ha='center', va='center', fontsize=8)
        else:
            plt.plot(y, x, 'o', color='red')

    plt.title("Input Data Mapped to SOM Grid")
    plt.xlim(-0.5, grid_size - 0.5)
    plt.ylim(-0.5, grid_size - 0.5)
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.xticks(range(grid_size))
    plt.yticks(range(grid_size))
    plt.savefig("grid.png")


if __name__=="__main__":
    
    with open(sys.argv[1],"r") as f:
        config=json.load(f)

    grid_size=config["grid_size"]
    df=pd.read_csv(config["data_source"],delimiter=',')
    radius=config["radius"]
    constant_radius=config["constant_radius"]
    learning_rate=config["learning_rate"]

    sim_function=get_metric_function(config["similarity_metric"])
    iterations=config["iterations"]

    countries=df["Country"]
    data=df.drop(columns=["Country"])
    data_scaled=(data-data.mean())/data.std(ddof=0)
    kohonen: Kohonen=Kohonen(grid_size, data_scaled, sim_function,radius, constant_radius)
    kohonen.train_network(iterations=iterations)

    plot_data_mapping(kohonen,countries)

