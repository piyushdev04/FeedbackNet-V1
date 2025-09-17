# Install required libraries
!pip install torch matplotlib --quiet

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

###########################
# 1. Multi-hop Synthetic Dataset
###########################

def generate_multihop_data(n_samples=10000):
    data = []
    for _ in range(n_samples):
        # Step 1: basic addition
        a = random.randint(1, 10)
        b = random.randint(1, 10)
        sum_ab = a + b
        if sum_ab > 20: sum_ab = 20

        # Step 2: subtraction
        c = random.randint(1, 5)
        remaining = sum_ab - c
        if remaining < 0: remaining = 0

        # Step 3: add another person's transfer
        d = random.randint(0, 5)
        final_total = remaining + d
        if final_total > 20: final_total = 20

        # Multi-hop question chaining
        question = f"Alice has {a} apples. Bob gives her {b}. She eats {c}. Carol gives her {d}. How many apples does Alice have?"
        answer = str(final_total)

        data.append((question, answer))
    return data

train_data = generate_multihop_data(10000)
test_data = generate_multihop_data(2000)

###########################
# 2. Tokenizer
###########################

class SimpleTokenizer:
    def __init__(self):
        self.vocab = {"<PAD>":0, "<UNK>":1}

    def encode(self, text):
        tokens = text.lower().split()
        ids = []
        for tok in tokens:
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)
            ids.append(self.vocab[tok])
        return ids

    def decode(self, ids):
        inv_vocab = {v:k for k,v in self.vocab.items()}
        return " ".join(inv_vocab.get(i, "<UNK>") for i in ids)

tokenizer = SimpleTokenizer()

def encode_dataset(dataset):
    X, y = [], []
    valid_count = 0
    for question, answer in dataset:
        tokenizer.encode(question)
        tokenizer.encode(answer)
    for question, answer in dataset:
        q_ids = tokenizer.encode(question)
        a_ids = tokenizer.encode(answer)
        if len(a_ids) == 1:  # single-token answers only
            X.append(torch.tensor(q_ids))
            y.append(torch.tensor(a_ids))
            valid_count += 1
    print(f"Valid samples (single-token answers): {valid_count}/{len(dataset)}")
    return X, y

X_train, y_train = encode_dataset(train_data)
X_test, y_test = encode_dataset(test_data)

print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

###########################
# 3. Padding & Dataset Preparation
###########################

def pad_sequence(seq, max_len, pad_value=0):
    padded = seq.tolist() + [pad_value] * (max_len - len(seq))
    return torch.tensor(padded)

def prepare_dataset(X, y):
    max_len = max(len(seq) for seq in X)
    X_padded = []
    for seq in X:
        X_padded.append(pad_sequence(seq, max_len))
    y_tensor = torch.stack(y).squeeze().long()
    X_tensor = torch.stack(X_padded)
    return X_tensor, y_tensor

X_train_tensor, y_train_tensor = prepare_dataset(X_train, y_train)
X_test_tensor, y_test_tensor = prepare_dataset(X_test, y_test)

print(f"Train tensor shape: {X_train_tensor.shape}")
print(f"Test tensor shape: {X_test_tensor.shape}")
print(f"Vocabulary size: {len(tokenizer.vocab)}")

###########################
# 4. Models
###########################

class FeedForwardTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=2, batch_first=False)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        attn_output, _ = self.attn(x, x, x)
        out = self.fc(attn_output[-1])
        return out

class CortexNet(nn.Module):
    def __init__(self, vocab_size, embed_dim=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=2, batch_first=False)
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.feedback_proj = nn.Linear(embed_dim, embed_dim)
        self.gate_proj = nn.Linear(embed_dim*2, embed_dim)

    def forward(self, x, feedback=None):
        x_embed = self.embed(x)
        attn_output, _ = self.attn(x_embed, x_embed, x_embed)

        if feedback is not None:
            feedback_proj = self.feedback_proj(feedback)
            combined = torch.cat([attn_output[-1], feedback_proj], dim=-1)
            gate = torch.sigmoid(self.gate_proj(combined))
            refined_output = (1 - gate) * attn_output[-1] + gate * feedback_proj
        else:
            refined_output = attn_output[-1]

        out = self.fc(refined_output)
        return out, refined_output

###########################
# 5. DataLoaders
###########################

BATCH_SIZE = 64
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=BATCH_SIZE)

###########################
# 6. Training and Evaluation
###########################

def train_model(model, data_loader, epochs=50, lr=0.01):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    losses = []

    for epoch in range(epochs):
        total_loss = 0
        for q, a in data_loader:
            q = q.to(device).transpose(0,1)
            a = a.to(device)
            optimizer.zero_grad()
            if isinstance(model, CortexNet):
                out, feedback = model(q)
                out_refined, _ = model(q, feedback=feedback)
                loss = criterion(out_refined, a)
            else:
                out = model(q)
                loss = criterion(out, a)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        losses.append(avg_loss)
    return losses

def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for q, a in data_loader:
            q = q.to(device).transpose(0,1)
            a = a.to(device)
            if isinstance(model, CortexNet):
                out, feedback = model(q)
                out_refined, _ = model(q, feedback=feedback)
                pred = out_refined.argmax(dim=-1)
            else:
                out = model(q)
                pred = out.argmax(dim=-1)
            correct += (pred == a).sum().item()
            total += a.size(0)
    acc = correct / total
    print(f"Accuracy: {acc*100:.2f}%")
    return acc

###########################
# 7. Run Experiments
###########################

EPOCHS = 50
LEARNING_RATE = 0.01
vocab_size = len(tokenizer.vocab)

# FeedForward Transformer
print("\nTraining FeedForward Transformer...")
ff_model = FeedForwardTransformer(vocab_size).to(device)
ff_losses = train_model(ff_model, train_loader, epochs=EPOCHS, lr=LEARNING_RATE)
ff_acc = evaluate_model(ff_model, test_loader)

# CortexNet Transformer
print("\nTraining FeedbackNet V1...")
cn_model = CortexNet(vocab_size).to(device)
cn_losses = train_model(cn_model, train_loader, epochs=EPOCHS, lr=LEARNING_RATE)
cn_acc = evaluate_model(cn_model, test_loader)

# Plot Loss Curves
plt.plot(ff_losses, label="FeedForward Transformer")
plt.plot(cn_losses, label="FeedbackNet V1 (Feedback Transformer)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.show()

# Plot Accuracy
plt.bar(["FeedForward", "FeedbackNet V1"], [ff_acc*100, cn_acc*100])
plt.ylabel("Accuracy (%)")
plt.title("Final Accuracy Comparison")
plt.show()