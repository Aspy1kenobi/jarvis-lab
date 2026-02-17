import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Self-Attention Layer
# ----------------------------
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, head_dim):
        """ 
        Single attention head.

        embed-dim: Size of input embeddings (e.g., 64)
        head_dim: Size of Q, K, V vectors (e.g. 16)
        """
        super().__init__()
        self.head_dim = head_dim

        # Linear projection for Q, K, V
        self.query = nn.Linear(embed_dim, head_dim, bias=False)
        self.key = nn.Linear(embed_dim, head_dim, bias=False)
        self.value = nn.Linear(embed_dim, head_dim, bias=False)

    def forward(self, x):
        """ 
        x: (batch_size, seq_len, embed_dim)

        Returns: (batch_size, seq_len, head_dim)
        """
        B, T, C = x.shape # Batch, Time (sequence), Channels (embedding)

        # Step 1: Create Q, K, V
        Q = self.query(x) # (B, T, head_dim)
        K = self.key(x) # (B, T, head_dim)
        V = self.value(x) # (B, T, head_dim)

        # Step 2: Compute attention scores
        # Q @ K^T gives us (B, T, T) - every position attending
        scores  = Q @ K.transpose(-2, -1) # (B, T, T)

        # Scales by sqrt(head_dim) for stability
        scores = scores / (self.head_dim ** 0.5)

        # Step 3: Apply causal mask (can't look into the future!)
        # This is CRITICAL for language modeling
        mask = torch.tril(torch.ones(T, T)) #Lower triangular matrix
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # Step 4: convert to probabbilities
        attention_weights = F.softmax(scores, dim=-1) # (B, T, T)

        # Step 5: Apply to values
        output = attention_weights @ V # (B, T, head_dim)

        return output
    
# ----------------------------
# Test It!
# ----------------------------

if __name__ == "__main__":
    # Create fake data
    batch_size = 2
    seq_len = 10 # Like "ROMEO: wha"
    embed_dim = 64

    # Random embeddings ( like from your embedding layer)
    x = torch.randn(batch_size, seq_len, embed_dim)

    # Create attention layer
    attn = SelfAttention(embed_dim=64, head_dim=16)

    # Run attention
    output = attn(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("\n✓ Attention works")

    # Let's visualize the attention pattern
    with torch.no_grad():
        Q = attn.query(x[0]) # Just first batch
        K = attn.key(x[0])
        scores = Q @ K.transpose(-1, -2) / (16 ** 0.5)

        # Apply mask
        mask = torch.tril(torch.ones(seq_len, seq_len))
        scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)

        print("\nAttention pattern (position i attending to position j):")
        print("Rows - query position, Cols = key position")
        print(weights.numpy().round(2))