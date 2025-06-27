from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def save_letter_heatmap(output_vector, filename, shape=(7, 5), cmap='gray', vmin=0, vmax=1,binarize=True):
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
    plt.imshow((data>0.5).astype(int) if binarize else data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def png_to_rgba_array(path: str, flatten: bool = True) -> np.ndarray:
    img = Image.open(path).convert("RGBA")
    arr: np.ndarray = np.asarray(img, dtype=np.float32)
    arr /= np.float32(255.0)
    if flatten:
        arr = arr.reshape(-1)
    return arr

def rgba_array_to_png(arr: np.ndarray, path: str, original_shape: Optional[Tuple[int, int]] = None) -> None:
    if arr.ndim == 1:
        if original_shape is None:
            raise ValueError(
                "Flat array provided — supply original_shape=(H, W) to reshape."
            )
        H, W = original_shape
        arr = arr.reshape((H, W, 4))

    arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)

    img = Image.fromarray(arr, mode="RGBA")
    img.save(path)
def save_merged_heatmaps(outputs, filename="merged.png", grid_shape=(6, 6), pad=2, threshold=0.5,binary=False):
    n_images = len(outputs)
    grid_h, grid_w = grid_shape

    # Reshape and binarize each output to (7, 5)
    reshaped_imgs = [(img.reshape((7, 5)) > threshold).astype(float) if binary else img.reshape((7,5)) for img in outputs]
    padded_imgs = [np.pad(img, pad_width=pad, mode='constant', constant_values=0) for img in reshaped_imgs]
    blank = np.zeros_like(padded_imgs[0])
    padded_outputs = padded_imgs + [blank] * (grid_h * grid_w - n_images)

    rows = []
    for i in range(grid_h):
        row_imgs = padded_outputs[i * grid_w:(i + 1) * grid_w]
        row = np.concatenate(row_imgs, axis=1)
        rows.append(row)
    grid_img = np.concatenate(rows, axis=0)

    plt.figure(figsize=(grid_w * 4, grid_h * 4))
    plt.axis('off')
    plt.imshow(grid_img, cmap='gray', vmin=0, vmax=1)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    