import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import re
import os
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 1. Load 10% of the dataset and tokenize
print("Loading dataset...")
ds = load_dataset("nirajandhakal/Mahabharata-HHGTTG-Text", split="train[:10%]")
corpus = " ".join(ds['text'])

def tokenize(text):
    return re.findall(r"\b\w+\b|[^\w\s]", text.lower())

tokens = tokenize(corpus)
print(f"Number of tokens: {len(tokens)}")
print("Sample tokens:", tokens[:20])

# 2. Build vocabulary and encode
vocab_size = 10000
most_common = Counter(tokens).most_common(vocab_size-2)
vocab = [w for w, _ in most_common]
word2idx = {w: i+2 for i, w in enumerate(vocab)}
word2idx["<PAD>"] = 0
word2idx["<UNK>"] = 1
idx2word = {i: w for w, i in word2idx.items()}

encoded = [word2idx.get(w, 1) for w in tokens]
print(f"Vocabulary size: {len(word2idx)}")

# 3. Prepare data, train/val split
seq_length = 15
step = 1

sequences = []
next_words = []
for i in range(0, len(encoded) - seq_length, step):
    sequences.append(encoded[i:i+seq_length])
    next_words.append(encoded[i+seq_length])

X = np.array(sequences, dtype=np.int32)
y = np.array(next_words, dtype=np.int32)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

from torch.utils.data import TensorDataset, DataLoader

batch_size = 256
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.long))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.long), torch.tensor(y_val, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# 4. Define the model
class WordLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=3, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden

embed_size = 256
hidden_size = 512
num_layers = 3
dropout = 0.3

model = WordLSTM(len(word2idx), embed_size, hidden_size, num_layers, dropout).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)

# 5. Training loop with loss & accuracy curves
epochs = 10
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

def accuracy(preds, targets):
    return (preds.argmax(dim=1) == targets).float().mean().item()

print("Starting training...")
for epoch in range(epochs):
    model.train()
    total_loss, total_acc, total_count = 0, 0, 0
    for xb, yb in tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{epochs}"):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        output, _ = model(xb)
        loss = loss_fn(output, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        total_acc += accuracy(output, yb) * xb.size(0)
        total_count += xb.size(0)
    avg_loss = total_loss / total_count
    avg_acc = total_acc / total_count
    train_losses.append(avg_loss)
    train_accuracies.append(avg_acc)

    # Validation
    model.eval()
    val_loss, val_acc, val_count = 0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            output, _ = model(xb)
            loss = loss_fn(output, yb)
            val_loss += loss.item() * xb.size(0)
            val_acc += accuracy(output, yb) * xb.size(0)
            val_count += xb.size(0)
    val_losses.append(val_loss / val_count)
    val_accuracies.append(val_acc / val_count)
    print(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Val Loss={val_losses[-1]:.4f}, Train Acc={avg_acc:.4f}, Val Acc={val_accuracies[-1]:.4f}")

# 6. Plot training & validation curves
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(train_accuracies, label='Train Acc')
plt.plot(val_accuracies, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("training_curves.png")
plt.show()
print("Training curves saved as training_curves.png")

# 7. Save the model
save_path = "word_lstm_standard.pth"
torch.save({
    'model_state_dict': model.state_dict(),
    'word2idx': word2idx,
    'idx2word': idx2word,
    'vocab_size': len(word2idx),
    'embed_size': embed_size,
    'hidden_size': hidden_size,
    'num_layers': num_layers,
    'dropout': dropout,
    'seq_length': seq_length
}, save_path)
print(f"Model saved to {save_path}")

# 8. Load the model
def load_model(path):
    checkpoint = torch.load(path, map_location=device)
    model = WordLSTM(
        checkpoint['vocab_size'],
        checkpoint['embed_size'],
        checkpoint['hidden_size'],
        checkpoint['num_layers'],
        checkpoint['dropout']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['word2idx'], checkpoint['idx2word'], checkpoint['seq_length']

loaded_model, loaded_word2idx, loaded_idx2word, loaded_seq_length = load_model(save_path)
print("Model loaded!")

# 9. Text generation
def generate_text(model, word2idx, idx2word, seq_length, seed, length=40, temperature=1.0):
    model.eval()
    def tokenize(text):
        return re.findall(r"\b\w+\b|[^\w\s]", text.lower())
    seed_tokens = tokenize(seed.lower())
    seed_encoded = [word2idx.get(w, 1) for w in seed_tokens]
    if len(seed_encoded) < seq_length:
        seed_encoded = [0]*(seq_length-len(seed_encoded)) + seed_encoded
    else:
        seed_encoded = seed_encoded[-seq_length:]
    generated = seed_tokens.copy()
    inp = torch.tensor([seed_encoded], dtype=torch.long, device=device)
    hidden = None
    for _ in range(length):
        out, hidden = model(inp, hidden)
        out = out[0].detach().cpu().numpy()
        out = out / temperature
        exp_out = np.exp(out - np.max(out))
        probs = exp_out / np.sum(exp_out)
        idx = np.random.choice(range(len(idx2word)), p=probs)
        next_word = idx2word.get(idx, "<UNK>")
        generated.append(next_word)
        inp = torch.cat([inp[:, 1:], torch.tensor([[idx]], device=device)], dim=1)
    return " ".join(generated)

seed_text = "The universe is"
print("\nGenerated text (temperature=0.8):\n")
print(generate_text(loaded_model, loaded_word2idx, loaded_idx2word, loaded_seq_length, seed_text, length=40, temperature=0.8))
