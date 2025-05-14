import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EncoderLayer(nn.Module):
    def __init__(self, d_model=64, n_heads=4, ff_dim=128):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads 
        self.head_dim = d_model // n_heads

        #attention projections
        self.qkv_proj = nn.Linear(d_model, d_model * 3) #efficient way of projecting to q, k, v 
        self.out_proj = nn.Linear(d_model, d_model)

        #FF MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model)
        )

        #layernorms 
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        B, N, D = x.shape 

        #multi-head attention
        x_norm = self.norm1(x)
        qkv = self.qkv_proj(x_norm)
        qkv = qkv.reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4) # qkv: (3, B, n_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, n_heads, N, N)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = attn_weights @ v  # (B, n_heads, N, head_dim)

        attn_output = attn_output.transpose(1, 2).reshape(B, N, D)  # (B, N, D)
        attn_output = self.out_proj(attn_output)
        x = x + attn_output  # Residual connection

        # === Feedforward ===
        x_norm = self.norm2(x)
        x = x + self.mlp(x_norm)  # Residual

        return x
    

class TransformerEncoder(nn.Module):
    def __init__(self, depth=4, d_model=64, n_heads=4, ff_dim=128, num_patches=16):
        super().__init__()

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, d_model))  # (1, 16, 64)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model=d_model, n_heads=n_heads, ff_dim=ff_dim)
            for _ in range(depth)
        ])

    def forward(self, x):
        """
        x: Tensor of shape (B, num_patches, d_model)
        returns: Tensor of same shape (B, num_patches, d_model)
        """
        x = x + self.pos_embedding  

        for layer in self.layers:
            x = layer(x)

        return x
    

# simple testing of dimensions 
if __name__ == "__main__":
    import torch

    B = 4  # batch size
    N = 16  # number of patches
    D = 64  # embedding dim

    dummy_input = torch.randn(B, N, D)

    print("Testing EncoderLayer...")
    layer = EncoderLayer(d_model=D, n_heads=4, ff_dim=128)
    out = layer(dummy_input)
    print("EncoderLayer output shape:", out.shape)  # (B, N, D) torch.Size([4, 16, 64])

    print("Testing TransformerEncoder...")
    encoder = TransformerEncoder(depth=3, d_model=D, n_heads=4, ff_dim=128, num_patches=N)
    out = encoder(dummy_input)
    print("TransformerEncoder output shape:", out.shape)  # (B, N, D) torch.Size([4, 16, 64])