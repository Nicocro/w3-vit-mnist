import numpy as np
import torch
from PIL import Image

def preprocess_canvases(images):
    """
    Takes a list of 4 RGBA images (top-left, top-right, bottom-left, bottom-right),
    resizes to 28x28, converts to grayscale, stitches to (1, 56, 56) tensor.
    """
    assert len(images) == 4, "Expected 4 images"

    digits = []
    for img in images:
        img = Image.fromarray(img).convert("L")  # convert to grayscale
        img = img.resize((28, 28))
        img = np.array(img).astype(np.float32) / 255.0  # scale to [0, 1]
        digits.append(img)

    top = np.hstack([digits[0], digits[1]])
    bottom = np.hstack([digits[2], digits[3]])
    grid = np.vstack([top, bottom])  # shape (56, 56)

    # Normalize like MNIST
    grid = (grid - 0.1307) / 0.3081
    grid = torch.tensor(grid).unsqueeze(0).unsqueeze(0)  # shape (1, 1, 56, 56)
    return grid