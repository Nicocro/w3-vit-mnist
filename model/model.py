import torch
import torch.nn as nn
from .feature_extractor import FeatureExtractor
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder

class ImageToDigitTransformer(nn.Module):
    def __init__(self, vocab_size=13, d_model=64, n_heads=4, ff_dim=128,
                 encoder_depth=4, decoder_depth=2, num_patches=16, max_seq_len=5):
        super().__init__()

        self.feature_extractor = FeatureExtractor(patch_size=14, emb_dim=d_model)
        self.encoder = TransformerEncoder(
            depth=encoder_depth,
            d_model=d_model,
            n_heads=n_heads,
            ff_dim=ff_dim,
            num_patches=num_patches
        )
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            max_len=max_seq_len,
            d_model=d_model,
            n_heads=n_heads,
            ff_dim=ff_dim,
            depth=decoder_depth
        )

    def forward(self, image_tensor, decoder_input_ids):
        """
        image_tensor: (B, 1, 56, 56)
        decoder_input_ids: (B, 5)
        Returns:
            logits: (B, 5, vocab_size)
        """
        patch_embeddings = self.feature_extractor(image_tensor)       # (B, 16, 64)
        encoder_output = self.encoder(patch_embeddings)               # (B, 16, 64)
        logits = self.decoder(decoder_input_ids, encoder_output)     # (B, 5, 13)
        return logits

if __name__ == '__main__':
    model = ImageToDigitTransformer()
    img = torch.randn(4, 1, 56, 56)
    tokens = torch.randint(0, 13, (4, 5))
    logits = model(img, tokens)
    print(logits.shape)  # Expected: (4, 5, 13)