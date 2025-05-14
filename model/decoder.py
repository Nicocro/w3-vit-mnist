import torch
import torch.nn as nn
import torch.nn.functional as F 
import math 

class DecoderLayer(nn.Module):
    def __init__(self, d_model=64, n_heads=4, ff_dim=128):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads 
        self.head_dim = d_model // n_heads 

        assert d_model % n_heads == 0, "d_model must be divisible by number of heads"

        # Self-attention: Q, K, V from decoder input
        self.self_attn_proj = nn.Linear(d_model, 3 * d_model)

        # Cross-attention: Q from decoder input, K/V from encoder output
        self.cross_attn_q = nn.Linear(d_model, d_model)
        self.cross_attn_kv = nn.Linear(d_model, 2 * d_model)

        # Output projections
        self.self_out = nn.Linear(d_model, d_model)
        self.cross_out = nn.Linear(d_model, d_model)

        # Feedforward MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model)
        )

        # LayerNorms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out):
            """
            x: (B, T, D) - decoder input embeddings
            enc_out: (B, N, D) - encoder outputs (image patch representations)
            Returns: (B, T, D)
            """
            B, T, D = x.shape
            _, N, _ = enc_out.shape

            # Masked Self-Attention 
            x_norm = self.norm1(x)
            qkv = self.self_attn_proj(x_norm).reshape(B, T, 3, self.n_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, T, head_dim)
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, n_heads, T, T)

            # Causal mask: prevent attention to future positions
            mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_out = attn_weights @ v  # (B, n_heads, T, head_dim)
            attn_out = attn_out.transpose(1, 2).reshape(B, T, D)
            attn_out = self.self_out(attn_out)
            x = x + attn_out  # Residual

            # Cross-Attention 
            x_norm = self.norm2(x)
            q = self.cross_attn_q(x_norm).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T, head_dim)
            kv = self.cross_attn_kv(enc_out).reshape(B, N, 2, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]  # (B, n_heads, N, head_dim)

            cross_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, n_heads, T, N)
            cross_weights = F.softmax(cross_scores, dim=-1)
            cross_out = cross_weights @ v  # (B, n_heads, T, head_dim)
            cross_out = cross_out.transpose(1, 2).reshape(B, T, D)
            cross_out = self.cross_out(cross_out)
            x = x + cross_out  # Residual

            # Feedforward 
            x_norm = self.norm3(x)
            x = x + self.mlp(x_norm)  # Residual

            return x


# implement the entire decoder 

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size=13, max_len=5, d_model=64, n_heads=4, ff_dim=128, depth=2):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))  # (1, 5, 64)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model=d_model, n_heads=n_heads, ff_dim=ff_dim)
            for _ in range(depth)
        ])

        self.output_proj = nn.Linear(d_model, vocab_size)  # Final projection to logits

    def forward(self, decoder_input_ids, encoder_output):
        """
        decoder_input_ids: (B, T) token IDs
        encoder_output: (B, N, d_model) from image encoder
        returns: logits over vocab, shape (B, T, vocab_size)
        """
        x = self.token_embedding(decoder_input_ids)  # (B, T, d_model)
        x = x + self.pos_embedding[:, :x.size(1), :]  # Add positional embedding

        for layer in self.layers:
            x = layer(x, encoder_output)  # (B, T, d_model)

        logits = self.output_proj(x)  # (B, T, vocab_size)
        return logits
     

# quick test

if __name__ == "__main__":
    decoder = TransformerDecoder()
    decoder_input = torch.randint(0, 13, (4, 5))  # (B=4, T=5)
    encoder_out = torch.randn(4, 16, 64)  # (B=4, N=16, D=64)

    logits = decoder(decoder_input, encoder_out)
    print("Logits shape:", logits.shape)  # (4, 5, 13)