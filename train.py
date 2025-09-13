import torch
import torch.nn as nn
from torch.optim import Adam
from model import SentimentLSTM
from data import train_loader, test_loader

# Hyperparamètres
vocab_size = 20000
embedding_dim = 100
hidden_dim = 128
lr = 0.001
num_epochs = 5



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SentimentLSTM(vocab_size, embedding_dim, hidden_dim).to(device)
criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=lr)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device).unsqueeze(1)  # [batch, 1]

        outputs = model(input_ids)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions = (outputs > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss:.4f} - Acc: {acc:.4f}")


torch.save(model.state_dict(), "sentiment_model.pth")