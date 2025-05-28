import json
import sys
from src.kohonen import Kohonen
from src.similarity_metrics import get_metric_function
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from collections import defaultdict

seed = 23902980

np.random.seed(seed)

def plot_u_matrix(som: Kohonen,sim_function):
    grid_size = som.grid_size
    weights = som.weights
    u_matrix = np.zeros((grid_size, grid_size))

    for i in range(grid_size):
        for j in range(grid_size):
            current_index = i * grid_size + j
            current_weight = weights[current_index]

            # Get neighbors: up, down, left, right
            neighbors = []
            if i > 0:
                neighbors.append(weights[(i-1)*grid_size + j])
            if i < grid_size - 1:
                neighbors.append(weights[(i+1)*grid_size + j])
            if j > 0:
                neighbors.append(weights[i*grid_size + (j-1)])
            if j < grid_size - 1:
                neighbors.append(weights[i*grid_size + (j+1)])

            # Average distance to neighbors
            dists = [np.linalg.norm(current_weight - n) for n in neighbors]
            u_matrix[i, j] = np.mean(dists)

    plt.figure(figsize=(6, 6))
    plt.imshow(u_matrix, cmap='viridis', origin='upper')
    plt.colorbar(label='Average Distance')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"avgDistance-{sim_function}-{grid_size}x{grid_size}")

def plot_data_mapping(som: Kohonen, labels, sim_function):
    grid_size = som.grid_size
    mapped = som.map_input()

    # Group labels per neuron
    neuron_labels = defaultdict(list)
    for idx, neuron_index in enumerate(mapped):
        neuron_labels[tuple(neuron_index)].append(str(labels[idx]))

    plt.figure(figsize=(8, 8))
    for (x, y), lbls in neuron_labels.items():
        text = ', '.join(lbls)  # or: f"{len(lbls)} items" for dense datasets
        plt.text(x, y, text, ha='center', va='center', fontsize=6, wrap=True)

    plt.xlim(-0.5, grid_size - 0.5)
    plt.ylim(-0.5, grid_size - 0.5)
    plt.grid(True)
    plt.xticks(range(grid_size))
    plt.yticks(range(grid_size))
    plt.tight_layout()
    plt.savefig(f"grid-{sim_function}-{grid_size}x{grid_size}.png")

def plot_weight_maps(som: Kohonen, feature_names: list[str], sim_function: str):
    weights_grid = som.get_weights_grid()  # Shape: (grid_size, grid_size, n_features)
    n_features = weights_grid.shape[2]
    grid_size = som.grid_size

    x, y = np.meshgrid(range(grid_size), range(grid_size))
    x = x.flatten()
    y = y.flatten()

    for i in range(n_features):
        values = weights_grid[:, :, i].flatten()
        vmin = np.min(values)
        vmax = np.max(values)
        vcenter = 0

        plt.figure(figsize=(6, 6))

        scatter = plt.scatter(
            x, y,
            c=values,
            cmap='RdYlGn',
            norm=TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax),
            s=800, marker='o', edgecolor='k', linewidth=0.5
        )

        plt.colorbar(scatter, label='Weight Value',
            location='left', fraction=0.046, pad=0.02)
        plt.title(f"{feature_names[i]}")
        plt.xticks([])
        plt.yticks([])
        plt.gca().set_aspect('equal')
        plt.xlim(-0.5, grid_size - 0.5)
        plt.ylim(-0.5, grid_size - 0.5)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f"weightmap-{feature_names[i]}-{sim_function}-{grid_size}x{grid_size}.png")
        plt.close()

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

    plot_data_mapping(kohonen,countries,config["similarity_metric"])
    plot_u_matrix(kohonen,config["similarity_metric"])
    plot_weight_maps(kohonen, list(data.columns), config["similarity_metric"])
