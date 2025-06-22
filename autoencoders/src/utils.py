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