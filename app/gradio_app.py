import sys
sys.path.append('.')

import gradio as gr
import torch
import numpy as np
from PIL import Image, ImageDraw

from model.model import ImageToDigitTransformer
from utils.tokenizer import START, FINISH, decode
from app.preprocess import preprocess_canvases

# Load model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = ImageToDigitTransformer(vocab_size=13).to(device)
model.load_state_dict(torch.load("checkpoints/transformer_mnist.pt", map_location=device))
model.eval()

def split_into_quadrants(image):
    """Split a PIL Image or numpy array into 4 quadrants (TL, TR, BL, BR)."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    w, h = image.size
    return [
        np.array(image.crop((0, 0, w // 2, h // 2))),
        np.array(image.crop((w // 2, 0, w, h // 2))),
        np.array(image.crop((0, h // 2, w // 2, h))),
        np.array(image.crop((w // 2, h // 2, w, h))),
    ]

def predict_digit_sequence(editor_data):
    """Predicts 4-digit sequence from 2×2 canvas image."""
    if editor_data is None or "composite" not in editor_data:
        return "No image provided."
    img = editor_data["composite"]
    quadrants = split_into_quadrants(img)
    image_tensor = preprocess_canvases(quadrants).to(device)

    decoded = [START]
    for _ in range(4):
        input_ids = torch.tensor(decoded, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(image_tensor, input_ids)
        next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
        decoded.append(next_token)
        if next_token == FINISH:
            break

    pred = decoded[1:]
    return "".join(decode(pred[:4]))

def create_black_canvas(size=(800, 800)):
    """Create a black canvas with a 2×2 light gray grid overlay."""
    img = Image.new("L", size, color=0)
    draw = ImageDraw.Draw(img)
    w, h = size
    draw.line([(w // 2, 0), (w // 2, h)], fill=128, width=2)
    draw.line([(0, h // 2), (w, h // 2)], fill=128, width=2)
    return img

# === UI ===
canvas_size = 800

with gr.Blocks() as demo:
    gr.Markdown("## Draw 4 digits in a 2×2 grid using a white brush")

    canvas = gr.ImageEditor(
        label="White brush only on black canvas (no uploads)",
        value=create_black_canvas(),
        image_mode="L",
        height=canvas_size,
        width=canvas_size,
        sources=[],  # disables uploads
        type="pil",
        brush=gr.Brush(colors=["#FFFFFF"], default_color="#FFFFFF", default_size=15, color_mode="fixed")
    )

    predict_btn = gr.Button("Predict")
    clear_btn = gr.Button("Erase")
    output = gr.Textbox(label="Predicted 4-digit sequence", interactive=True)

    predict_btn.click(fn=predict_digit_sequence, inputs=[canvas], outputs=[output])
    clear_btn.click(fn=lambda: create_black_canvas(), inputs=[], outputs=[canvas])

demo.launch()