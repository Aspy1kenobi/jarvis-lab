import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# ----------------------------
# Load Training Data
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
print("First 200 chars:\n", text[:200])

# ----------------------------
# Split Data 
# ----------------------------
split_idx = int(0.9 * len(data))
train_data = data[:split_idx]
val_data = data[split_idx:]
print(f"\nTrain set: {len(train_data)} chars")
print(f"Val set: {len(val_data)} chars")


# ----------------------------
# Hyperparameters
# ----------------------------

block_size = 64
batch_size = 32
embedding_dim = 128
hidden_dim = 512
learning_rate = 3e-4
epochs = 10000

# ----------------------------
# Create Batch Function
# ----------------------------

def get_batch(split='train'):
    """Get a batch from either train or validation data"""
    source_data = train_data if split == 'train' else val_data
    ix = torch.randint(len(source_data) - block_size, (batch_size,))
    x = torch.stack([source_data[i:i+block_size] for i in ix])
    y = torch.stack([source_data[i+1:i+block_size+1] for i in ix])
    return x, y

# ----------------------------
# Model Definition
# ----------------------------

class TinyLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, block_size):
        super().__init__()
        self.block_size = block_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Deeper network with multiple layers
        self.linear1 = nn.Linear(embedding_dim * block_size, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

        #Dropout for regularization (prevents overfitting)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x shape: (batch_size, block_size)
        x = self.embedding(x)              # (B, T, C)
        x = x.view(x.size(0), -1)         # Flatten to (B, T*C)
        
        # Layer 1
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        
        # Layer 2
        x = F.relu(self.linear2(x))
        x = self.dropout(x)
        
        # Layer 3
        x = F.relu(self.linear3(x))
        x = self.dropout(x)
        
        # Output layer
        logits = self.output(x)
        return logits

model = TinyLM(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    block_size=block_size
)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Learning rate scheduler - reduces LR when stuck
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=30,
    min_lr=1e-6
)

print("✓ Learning rate scheduler enabled")
print(f"\nModel Parameters:")
print(f" Block size: {block_size}")
print(f" Embedding dim: {embedding_dim}")
print(f" Hidden dim: {hidden_dim}")
print(f" Total params: {sum(p.numel() for p in model.parameters()):,}")

# ----------------------------
# Create models directory (ADD THIS)
# ----------------------------
os.makedirs("models", exist_ok=True)


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
    logits = model(xb)
    loss = F.cross_entropy(logits, yb[:, -1])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        model.eval()
        with torch.no_grad():
            # Compute validation loss
            val_xb, val_yb = get_batch('val')
            val_logits = model(val_xb)
            val_loss = F.cross_entropy(val_logits, val_yb[:, -1])
        
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
                'vocab_size': vocab_size,
                'stoi': stoi,
                'itos': itos,
                'block_size': block_size,
                'embedding_dim': embedding_dim,
                'hidden_dim': hidden_dim,

            }, "models/tiny_lm_best.pt")
            print(f" → Saved new best model (val loss: {val_loss.item():.4f})")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"\n✓ Early stopping at epoch {epoch} (no improvement for {patience} checks)")
            print(f"No improvement for {patience} validation checks")
            print(f"Best validation loss: {best_val_loss:.4f}")
            break
        # Step the scheduler
        scheduler.step(val_loss)
# ----------------------------
# Save Final Model (ADD THIS)
# ----------------------------

torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'vocab_size': vocab_size,
    'stoi': stoi,
    'itos': itos,
}, "models/tiny_lm_final.pt")
print("\n✓ Saved final model to models/tiny_lm_final.pt")


# ----------------------------
# Text Generation
# ----------------------------

def generate(model, start_text, length=200, temperature=1.0):
    model.eval()
    context = [stoi[c] for c in start_text]

    for _ in range(length):
        # Pad if needed
        if len(context) < block_size:
            padded = [0] * (block_size - len(context)) + context
        else:
            padded = context[-block_size:]

        x = torch.tensor(padded).unsqueeze(0)
        logits = model(x)

        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1).item()
        context.append(next_idx)
    return "".join([itos[i] for i in context])

print("\nGenerated Text (temperature=1.0):\n")
print(generate(model, start_text="hello ", length=200, temperature=1.0))

print("\nGenerated Text (temperature=0.8 - more focused):\n")
print(generate(model, start_text="hello ", length=200, temperature=0.8))