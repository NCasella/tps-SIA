import colorsys
import json
import os
import random
import sys
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator
from src.kohonen import Kohonen
from src.similarity_metrics import get_metric_function
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from collections import defaultdict

seed = 23902980

np.random.seed(seed)

def plot_u_matrix(som: Kohonen, sim_function: str):
    grid_size = som.grid_size
    weights = som.weights
    u_matrix = np.zeros((grid_size, grid_size))

    for i in range(grid_size):
        for j in range(grid_size):
            current_index = i * grid_size + j
            current_weight = weights[current_index]

            neighbors = []
            if i > 0:
                neighbors.append(weights[(i - 1) * grid_size + j])
            if i < grid_size - 1:
                neighbors.append(weights[(i + 1) * grid_size + j])
            if j > 0:
                neighbors.append(weights[i * grid_size + (j - 1)])
            if j < grid_size - 1:
                neighbors.append(weights[i * grid_size + (j + 1)])

            dists = [np.linalg.norm(current_weight - n) for n in neighbors]
            u_matrix[i, j] = np.mean(dists)

    x, y = np.meshgrid(range(grid_size), range(grid_size))
    x = x.flatten()
    y = y.flatten()
    values = u_matrix.flatten()

    vmin = np.min(values)
    vmax = np.max(values)
    vcenter = (vmax - vmin) / 2

    plt.figure(figsize=(6, 6))

    scatter = plt.scatter(
        x, y,
        c=values,
        cmap='RdYlGn',
        norm=TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax),
        s=800, marker='o', edgecolor='k', linewidth=0.5
    )

    plt.colorbar(scatter, label='Average Distance',
                 location='left', fraction=0.046, pad=0.02)
    plt.title('Unified Distance Matrix')
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_aspect('equal')
    plt.xlim(-0.5, grid_size - 0.5)
    plt.ylim(-0.5, grid_size - 0.5)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'output/avg-distance-{sim_function}-{grid_size}x{grid_size}.png')
    plt.close()


def plot_register_counts(som: Kohonen, sim_function: str):
    grid_size = som.grid_size
    mapped = som.map_input()

    counts = np.zeros((grid_size, grid_size), dtype=int)
    for x, y in mapped:
        counts[x, y] += 1

    max_count = counts.max()
    num_colors = max_count + 1 if max_count > 0 else 1

    base_cmap = plt.get_cmap('coolwarm')
    color_list = [base_cmap(i / (num_colors - 1)) for i in range(num_colors)]
    cmap = ListedColormap(color_list)

    x, y = np.meshgrid(range(grid_size), range(grid_size))
    x = x.flatten()
    y = y.flatten()
    values = counts.flatten()

    plt.figure(figsize=(6, 6))

    scatter = plt.scatter(
        x, y,
        c=values,
        cmap=cmap,
        s=800, marker='o', edgecolor='k', linewidth=0.5,
        vmin=-0.5, vmax=max_count + 0.5
    )

    cbar = plt.colorbar(scatter, location='left', fraction=0.046, pad=0.02)
    cbar.set_label('Count')
    cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.title('Register Counts')
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_aspect('equal')
    plt.xlim(-0.5, grid_size - 0.5)
    plt.ylim(-0.5, grid_size - 0.5)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'output/register-counts-{sim_function}-{grid_size}x{grid_size}.png')
    plt.close()


def plot_data_mapping(som: Kohonen, labels, sim_function):
    grid_size = som.grid_size
    mapped = som.map_input()

    plt.figure(figsize=(10, 10))
    grouped = defaultdict(list)
    color_dict = {}

    for idx, neuron_index in enumerate(mapped):
        x, y = neuron_index
        if labels is not None:
            grouped[(x, y)].append(str(labels[idx]))

    for coord in grouped.keys():
        if coord not in color_dict:
            h = random.random()
            s = 0.7 + random.random() * 0.3
            v = 0.3 + random.random() * 0.3
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            color_dict[coord] = (r, g, b)

    for (x, y), cluster_labels in grouped.items():
        text_content = '\n'.join(sorted(set(cluster_labels)))

        plt.text(x, y, text_content,
                 ha='center', va='center',
                 fontsize=9,
                 color='white',
                 bbox=dict(
                     boxstyle=f'round,pad=0.3',
                     facecolor=color_dict[(x, y)],
                     edgecolor='black',
                     alpha=0.95,
                     linewidth=0.8
                 ))

    plt.xlim(-1, grid_size)
    plt.ylim(-1, grid_size)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(grid_size))
    plt.yticks(range(grid_size))
    plt.savefig(f'output/grid-{sim_function}-{grid_size}x{grid_size}.png', dpi=130, bbox_inches='tight')
    plt.close()


def plot_weight_maps(som: Kohonen, feature_names: list[str], sim_function: str):
    weights_grid = som.get_weights_grid()
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
            norm=TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax),
            s=800, marker='o', edgecolor='k', linewidth=0.5
        )

        plt.colorbar(scatter, label='Weight Value',
                     location='left', fraction=0.046, pad=0.02)
        plt.title(f'{feature_names[i]}')
        plt.xticks([])
        plt.yticks([])
        plt.gca().set_aspect('equal')
        plt.xlim(-0.5, grid_size - 0.5)
        plt.ylim(-0.5, grid_size - 0.5)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'output/weightmap-{feature_names[i]}-{sim_function}-{grid_size}x{grid_size}.png')
        plt.close()


if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        config = json.load(f)

    os.makedirs('output', exist_ok=True)

    grid_size = config['grid_size']
    df = pd.read_csv(config['data_source'], delimiter=',')
    radius = config['radius']
    constant_radius = config['constant_radius']
    learning_rate = config['learning_rate']

    sim_function = get_metric_function(config['similarity_metric'])
    iterations = config['iterations']

    countries = df['Country']
    data = df.drop(columns=['Country'])
    data_scaled = (data - data.mean()) / data.std(ddof=0)
    data_scaled_np = data_scaled.to_numpy()
    initial_weights = None if config['random_weights'] else data_scaled_np[
        np.random.choice(len(data_scaled), size=grid_size ** 2)]
    kohonen: Kohonen = Kohonen(grid_size, data_scaled, sim_function, radius, constant_radius, initial_weights)
    kohonen.train_network(iterations=iterations)

    sim = config['similarity_metric']

    plot_data_mapping(kohonen, countries, sim)
    plot_u_matrix(kohonen, sim)
    plot_register_counts(kohonen, sim)
    plot_weight_maps(kohonen, list(data.columns), sim)
