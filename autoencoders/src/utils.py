import numpy as np
import matplotlib.pyplot as plt




def save_letter_heatmap(output_vector, filename, shape=(7, 5), cmap='gray', vmin=0, vmax=1):
    """
    Save a letter pattern as a heatmap image.

    Parameters:
    - output_vector: numpy array of shape (35,) or (1, 35)
    - filename: path to save the heatmap image
    - shape: reshape to this (rows, cols), default 7x5
    - cmap: matplotlib colormap (default: 'gray')
    - vmin, vmax: color scale range (useful for sigmoid/tanh outputs)
    """

    data = output_vector.reshape(shape)

    plt.figure(figsize=(2.5, 3.5))
    plt.imshow((data>0.5).astype(int), cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()