import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os

from dataset import MNIST_2x2
from model.model import ImageToDigitTransformer

# Use MPS if available (Apple Silicon)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Config
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
VOCAB_SIZE = 13

# Transforms 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Dataset & DataLoader 
train_base = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_dataset = MNIST_2x2(train_base, seed=42)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model, Loss, Optimizer 
model = ImageToDigitTransformer(vocab_size=VOCAB_SIZE).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training Loop
model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

    for images, dec_input, dec_target in loop:
        images = images.to(device)
        dec_input = dec_input.to(device)
        dec_target = dec_target.to(device)

        logits = model(images, dec_input)
        loss = loss_fn(logits.view(-1, VOCAB_SIZE), dec_target.view(-1))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

        # Update tqdm every batch
        loop.set_postfix(batch_loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {avg_loss:.4f}")

# save weights
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/transformer_mnist.pt")
print("Model saved to checkpoints/transformer_mnist.pt")