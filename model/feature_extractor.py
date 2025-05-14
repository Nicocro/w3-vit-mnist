import torch.nn as nn
import torch

class FeatureExtractor(nn.Module):
    def __init__(self, patch_size=14, emb_dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.proj = nn.Linear(patch_size * patch_size, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (B, 1, 56, 56)
        returns patch_embeddings of shape (B, 16, emb_dim)"""

        B, C, H, W = x.shape 
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(B, -1, self.patch_size * self.patch_size)
        patch_embeddings = self.proj(patches)

        return patch_embeddings



if __name__ == "__main__":

    feature_extractor = FeatureExtractor()
    dummy_input = torch.randn(8, 1, 56, 56)
    out = feature_extractor(dummy_input)

    print(out.shape) # should expect (8, 16, 64) 