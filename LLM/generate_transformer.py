import torch
import torch.nn.functional as F
from mini_transformer import MiniTransformer

def load_transformer(model_path="models/mini_transformer_best.pt"):
    """Load trained transformer"""
    checkpoint = torch.load(model_path)

    config = checkpoint['config']
    stoi = checkpoint['stoi']
    itos = checkpoint['itos']

    model = MiniTransformer(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ Loaded transformer from {model_path}")
    print(f"  Trained for {checkpoint['epoch']} epochs")
    print(f"  Validation loss: {checkpoint['val_loss']:.4f}")

    return model, stoi, itos, config['block_size']

def generate(model, stoi, itos, prompt, length=300, temperature=0.7, block_size=64):
    """Generate text from transformer"""
    model.eval()
    
    # Encode prompt
    context = [stoi.get(c, 0) for c in prompt]
    context = torch.tensor(context).unsqueeze(0)  # (1, T)
    
    with torch.no_grad():
        for _ in range(length):
            # Crop context to block_size
            context_crop = context[:, -block_size:]
            
            # Get predictions
            logits = model(context_crop)  # (1, T, vocab_size)
            logits = logits[:, -1, :]  # Take last position (1, vocab_size)
            
            # Apply temperature
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            
            # Sample
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to context
            context = torch.cat([context, next_token], dim=1)
    
    # Decode
    tokens = context[0].tolist()
    text = ''.join([itos[i] for i in tokens])
    return text

if __name__ == "__main__":
    # Load model
    model, stoi, itos, block_size = load_transformer()
    
    print("\n" + "="*60)
    print("MINI-TRANSFORMER vs FEEDFORWARD COMPARISON")
    print("="*60)
    
    prompts = ["ROMEO:", "First Citizen:"]
    
    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"Prompt: '{prompt}'")
        print('='*60)
        
        print("\nTemperature 0.7:")
        text = generate(model, stoi, itos, prompt, length=250, temperature=0.7, block_size=block_size)
        print(text)