---
title: MNIST 2x2 Transformer Demo
emoji: 🔢
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 5.29.0  # or your current gradio version
app_file: app.py
pinned: false
---

# Transformer MNIST 2×2 — Image-to-Sequence Prediction

In this project I implement a minimal Transformer-based model that takes a 2×2 grid of MNIST digits as input and autoregressively predicts the corresponding 4-digit sequence. It serves as a practical deep dive into the inner workings of the Transformer architecture and basic multimodality concepts, combining vision (image patches) with language modeling (digit sequences). Really enjoyed this one, forces you to think through transformer blocks quite deeply. 

## 1. Project Overview

The goal is to understand how a vanilla Transformer encoder-decoder can be applied to a simple multimodal task: mapping an image input to a discrete token sequence. This project focuses on building each architectural component from scratch and wiring them together cleanly.

## 2. Task Definition

- **Input:** a 2×2 grid composed of 4 random MNIST digits, forming a 56×56 grayscale image.
- **Output:** the 4-digit sequence corresponding to the digits in the grid (top-left → bottom-right).
- **Modeling approach:** sequence-to-sequence using an autoregressive decoder with special `<start>` and `<finish>` tokens.

## 3. Model Architecture

The model follows a clean encoder-decoder Transformer architecture:

- **Feature Extractor:** splits the 56×56 image into 16 non-overlapping patches of 14×14 pixels and projects each to a 64-dimensional embedding.
- **Transformer Encoder:** processes the 16 patch embeddings using standard multi-head self-attention, positional embeddings, and MLP blocks.
- **Transformer Decoder:** autoregressively predicts the digit sequence:
  - Uses masked self-attention over token embeddings
  - Attends to encoder output via cross-attention
  - Outputs a sequence of logits over a vocabulary of 13 tokens (digits 0–9, `<start>`, `<finish>`)
- **Tokenizer:** handles token ↔ digit conversions and input preparation.

## 4. Training Setup

- **Dataset:** MNIST, wrapped into a custom `MNIST_2x2` PyTorch dataset that returns the stitched image and 4-digit target.
- **Batch size:** 64
- **Epochs:** 10
- **Loss:** `CrossEntropyLoss` over vocabulary tokens
- **Optimizer:** Adam
- **Hardware:** Apple M4 with `mps` acceleration
- **Logging:** `tqdm` per-batch loss tracking for clear training progress

## 5. Evaluation

Evaluation is done on the held-out MNIST test set using greedy decoding:

- Starts with <start> token
- Predicts one token at a time (no teacher forcing)
- Stops after 4 tokens or if <finish> is predicted

### Evaluation Metrics

- **Sequence accuracy:** % of samples where all 4 digits are predicted correctly
- **Per-digit accuracy:** % of individual digits predicted correctly across all positions

### final results after 10 epochs of training

- **training loss at epoch 10:** 0.0101
- **Sequence accuracy:** 93.77% on held-out test set
- **Per digit accuracy:** 98.43% on held-out test set

## 6. Gradio App

Finally, I build a simple gradio app to let users draw digits on a 2X2 canvas grid with a brush.
After users submits, Inference runs and prints out the 4 predicted digits. 

The app is also hosted on a [Hugging Face Space](https://huggingface.co/spaces/nico-x/transformer-mnist-demo)

If you try it out you'll realise results are not that great, especially if you consider the high accuracy on
the test set. This is probably due to the out of distributon nature of the digit manually drawn by users with mouse on a canvas, which visibly differ from how digits are usually written inn MNIST. 

[![Watch the demo](https://cdn.loom.com/sessions/thumbnails/02080a8a466844df9fe6b55cd3d12c1a-bc0dbd9369660461-full-play.gif)](https://www.loom.com/share/02080a8a466844df9fe6b55cd3d12c1a?sid=f40a265a-39d7-41fd-8079-98b40a7d00b3)

Future work could include a pipeline to get feedback from users on good/bad outcomes and provide corrections that I could use as labels to trigger fine-tuning of the model.