import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_block import TransformerBlock

# ----------------------------
# Mini-Transformer Language Model
# ----------------------------
class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_dim, block_size):
        """
        A small transformer language model.

        vocab_size: Number of unique characters (e.g., 65)
        embed_dim: Embedding dimension (e.g., 64)
        num_heads: Number of attention heads (e.g., 4)
        num_layers: Number of transformer blocks (e.g., 4)
        ff_dim: Feed-forward hidden dim (e.g., 256)
        block_size: Context length (e.g., 64)
        """
        super().__init__()
        self.block_size = block_size

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # Positional embeddings (learned)
        self.position_embedding = nn.Embedding(block_size, embed_dim)

        # Stack of transformer block
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)]
        )

        # Final layer norm
        self.ln_f = nn.LayerNorm(embed_dim)

        # Output head ( predict next token)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

        # Dropout
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        x: (B, T) - batch of token indices

        Returns: (B, T, vocab_size) - logits for next token
        """
        B, T = x.shape

        # Get token embeddings
        tok_emb = self.token_embedding(x) # (B, T, embed_dim)

        # Get position embeddings
        pos = torch.arange(T, device=x.device) # (T,)
        pos_emb = self.position_embedding(pos) # (T, embed_dim)

        # Add them together
        x = self.dropout(tok_emb + pos_emb) # (B, T, embed_dim)

        # Pass through transformer blocks
        x = self.blocks(x) # (B, T, embed_dim)

        # Final layer norm
        x = self.ln_f(x) # (B, T, vocab_size)

        # Predict next tokens
        logits = self.lm_head(x) # (B, T, vocab_size)

        return logits
    
# ----------------------------
# Test It!
# ----------------------------
if __name__ == "__main__":
    print("Building Mini-Transformer\n")

    # Model configuration (same size as your feedforward modelfor fair comparison)
    config = {
        'vocab_size': 65,
        'embed_dim': 64,
        'num_heads': 4,
        'num_layers': 4, # 4 transformer blocks
        'ff_dim': 256,
        'block_size': 64
    }    

    # Create model
    model = MiniTransformer(**config)

    # Test with dummy input
    batch_size = 4
    seq_len = 64
    x = torch.randint(0, config['vocab_size'], (batch_size, seq_len))

    # Forward pass
    logits = model(x)

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    print("\n✓ Mini-Transformer ready to train!")
    
    # Compare to your feedforward model
    print("\nComparison to Feedforward Model:")
    print("  Feedforward:     1,201,281 parameters")
    print(f"  Mini-Transformer: {total_params:,} parameters")
    print(f"  Difference:      {1201281 - total_params:,} fewer parameters!")
    print("\nYet transformers typically perform BETTER with fewer params!")