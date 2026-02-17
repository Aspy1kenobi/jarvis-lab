import torch
import torch.nn as nn
import torch.nn.functional as F

# Import our multi-head attention
from multi_head_attention import MultiHeadAttention

# ----------------------------
# Feed-Forward Network
# ----------------------------
class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        """
        Simple 2-layer MLP.

        embed_dim: Input/output size
        ff_dim: Hidden layer size (usually 4x embed_dim)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
    
    def forward(self, x):
        return self.net(x)
    
# ----------------------------
# Transformer Block
# ----------------------------
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        """
        One transformer block.

        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        ff_dim: Feed-forward hidden dimension
        """
        super().__init__()

        # Multi-head attention
        self.attention = MultiHeadAttention(embed_dim, num_heads)

        # Feed-forward network
        self.ff = FeedForward(embed_dim, ff_dim)

        # Layer normalization (for training stability)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        x: (B, T, embed_dim)

        Uses residual connection: output = x = sublayer(x)
        """

        # Attention block with residual
        attn_out = self.attention(self.ln1(x))
        x = x + self.dropout(attn_out) # type: ignore # Residual connection

        # Feed-forward block with residual
        ff_out = self.ff(self.ln2(x))
        x = x + self.dropout(ff_out) # Residual connection

        return x

# ----------------------------
# Test It!
# ----------------------------
if __name__ == "__main__":
    print("Testing Transformer Block\n")

    batch_size = 2
    seq_len = 10
    embed_dim = 64
    num_heads = 4
    ff_dim = 256 # 4x embed_dim is standard

    x = torch.randn(batch_size, seq_len, embed_dim)

    block = TransformerBlock(embed_dim, num_heads, ff_dim)

    output = block(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\n✓ Transformer block works!")

    # Count parameters
    total_params = sum(p.numel() for p in block.parameters())
    print(f"Total parameters: {total_params,}")

    print("\nComponents:")
    print(f"  • Multi-head attention ({num_heads} heads)")
    print(f"  • Feed-forward network (64 → 256 → 64)")
    print(f"  • Layer normalization (2 layers)")
    print(f"  • Residual connections (2)")