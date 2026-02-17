import torch
import torch.nn as nn
import torch.nn.functional as F
from mini_transformer import MiniTransformer
from datetime import datetime

# ----------------------------
# Load Data (same as before)
# ----------------------------
with open("data/train.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

print("Vocab size:", vocab_size)
print("Dataset length:", len(text))

# Split data
split_idx = int(0.9 * len(data))
train_data = data[:split_idx]
val_data = data[split_idx:]

# ----------------------------
# Hyperparameters
# ----------------------------
block_size =64
batch_size = 32
learning_rate = 3e-4
epochs = 3000

# Model config
model_config = {
    'vocab_size': vocab_size,
    'embed_dim': 128,
    'num_heads': 8,
    'num_layers': 6,
    'ff_dim': 512,
    'block_size': block_size
}

# ----------------------------
# Batch Function
# ----------------------------
def get_batch(split='train'):
    source_data = train_data if split == 'train' else val_data
    ix = torch.randint(len(source_data) - block_size - 1, (batch_size,))  # -1 for target
    x = torch.stack([source_data[i:i+block_size] for i in ix])
    y = torch.stack([source_data[i+1:i+block_size+1] for i in ix])
    return x, y

# ----------------------------
# Create Model
# ----------------------------
model = MiniTransformer(**model_config)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=30, min_lr=1e-6
)

print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters())}")
print("Starting training...\n")

# ----------------------------
# Training Loop
# ----------------------------
best_val_loss = float('inf')
patience = 100
patience_counter = 0

for epoch in range(epochs):
    # Training
    model.train()
    xb, yb = get_batch('train')
    
    logits = model(xb)  # (B, T, vocab_size)
    
    # For transformer, we predict at ALL positions, not just the last one
    # Reshape for cross entropy: (B*T, vocab_size) and (B*T)
    B, T, C = logits.shape
    logits = logits.reshape(B*T, C)
    targets = yb.reshape(B*T)  # Changed from .view() to .reshape()
    
    loss = F.cross_entropy(logits, targets)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Validation
    if epoch % 20 == 0:
        model.eval()
        with torch.no_grad():
            val_xb, val_yb = get_batch('val')
            val_logits = model(val_xb)
            
            B, T, C = val_logits.shape
            val_logits = val_logits.reshape(B*T, C)
            val_targets = val_yb.reshape(B*T)
            
            val_loss = F.cross_entropy(val_logits, val_targets)
        
        print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': model_config,
                'stoi': stoi,
                'itos': itos,
            }, "models/mini_transformer_best.pt")
            print(f" → Saved new best model (val loss: {val_loss.item():.4f})")
        else:
            patience_counter += 1
            print(f"   (No improvement: {patience_counter}/{patience})")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n✓ Early stopping at epoch {epoch}")
            print(f"   Best validation loss: {best_val_loss:.4f}")
            break
        
        scheduler.step(val_loss)

print("\n✓ Training complete!")