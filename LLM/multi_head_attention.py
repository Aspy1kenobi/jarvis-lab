import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Single Attention Head (from before)
# ----------------------------

class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.head_dim = head_dim
        
        self.query = nn.Linear(embed_dim, head_dim, bias=False)
        self.key = nn.Linear(embed_dim, head_dim, bias=False)
        self.value = nn.Linear(embed_dim, head_dim, bias=False)
    
    def forward(self, x):
        B, T, C = x.shape
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        scores = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)
        
        # Causal mask
        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        output = attention_weights @ V
        
        return output


# ----------------------------
# Multi-Head Attention (NEW!)
# ----------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        Run multiple attention heads in parallel.

        embed_dim: Size of input (e.g., 64)
        num_heads: Number of parallel heads (e.g., 4)
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads # Each head gets a portion

        # Create multiple heads
        self.heads = nn.ModuleList([
            AttentionHead(embed_dim, self.head_dim)
            for _ in range(num_heads)
        ])

        # Output projection (combines all heads)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        x: (B, T, embed_dim)

        Returns: (B, T, embed_dim)
        """
        # Run all heads in parallel
        head_outputs = [head(x) for head in self.heads] # List of (B, T, head_dim)

        # Concatenate along the embedding dimension
        out = torch.cat(head_outputs, dim=-1) # (B, T, embed_dim)

        # Final projection
        out = self.proj(out)

        return out

# ----------------------------
# Test It!
# ----------------------------
if __name__ == "__main__":
    print("Testing Multi-Head Attention\n")

    # Parameters
    batch_size = 2
    seq_len = 10
    embed_dim = 64
    num_heads = 4

    # Create input
    x = torch.randn(batch_size, seq_len, embed_dim)

    # Create multi-head attention
    mha = MultiHeadAttention(embed_dim=64, num_heads=4)

    # Run it
    output = mha(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\n✓ Multi-head attention works!")
    print(f"  • {num_heads} heads running in parallel")
    print(f"  • Each head has dimension {mha.head_dim}")
    print(f"  • Combined output: {embed_dim} dimensions")

    # Count parameters
    total_params = sum(p.numel() for p in mha.parameters())
    print (f"\nTotal parameters: {total_params:,}")
