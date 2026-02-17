import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Model Definition (Same as training)
# ----------------------------
class TinyLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=512, block_size=64):
        super().__init__()
        self.block_size = block_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)


        self.linear1 = nn.Linear(embedding_dim * block_size, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        
        x = F.relu(self.linear2(x))
        x = self.dropout(x)
        
        x = F.relu(self.linear3(x))
        x = self.dropout(x)
        
        logits = self.output(x)
        return logits

# ----------------------------
# Load Model
# ----------------------------
def load_model(model_path="models/tiny_lm_best.pt"):
    """Load a saved model"""
    checkpoint = torch.load(model_path)
    
    # Extract all needed info
    vocab_size = checkpoint['vocab_size']
    block_size = checkpoint.get('block_size', 64)
    embedding_dim = checkpoint.get('embedding_dim', 128)
    hidden_dim = checkpoint.get('hidden_dim', 512)
    stoi = checkpoint['stoi']
    itos = checkpoint['itos']
    
    # Create model with correct architecture
    model = TinyLM(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        block_size=block_size
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Loaded model from {model_path}")
    print(f"  Trained for {checkpoint['epoch']} epochs")
    print(f"  Validation loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    print(f"  Block size: {block_size}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, stoi, itos, block_size

# ----------------------------
# Generate Function
# ----------------------------
def generate(model, stoi, itos, start_text, length=200, temperature=1.0, block_size=64):
    """Generate text from a loaded model"""
    model.eval()
    context = [stoi.get(c, 0) for c in start_text]  # Use .get() for safety
    
    with torch.no_grad():
        for _ in range(length):
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

# ----------------------------
# Main Usage
# ----------------------------
if __name__ == "__main__":
    # Load the model
    model, stoi, itos, block_size = load_model("models/tiny_lm_best.pt")
    
    print("\nGenerating text with different temperatures...")
    print("=" * 60)
    
    temperatures = [0.5, 0.7, 0.9, 1.1]
    prompts = ["ROMEO:", "First ", "The king "]
    
    for temp in temperatures:
        print(f"\n{'='*60}")
        print(f"TEMPERATURE: {temp}")
        print('='*60)
        
        for prompt in prompts:
            print(f"\nPrompt: '{prompt}'")
            text = generate(model, stoi, itos, prompt, length=200, temperature=temp, block_size=block_size)
            print(text)
            print("-" * 40)