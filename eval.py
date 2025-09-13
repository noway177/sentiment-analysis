import torch
from model import SentimentLSTM
from data import test_loader, vocab_size  # tu peux aussi redéfinir vocab_size ici si besoin

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparamètres (doivent être les mêmes que ceux du modèle entraîné)
embedding_dim = 100
hidden_dim = 128

# 1. Charger le modèle
model = SentimentLSTM(vocab_size, embedding_dim, hidden_dim)
model.load_state_dict(torch.load("sentiment_model.pth", map_location=device))
model.to(device)

# 2. Mode évaluation
model.eval()

# 3. Évaluation
correct = 0
total = 0

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device).unsqueeze(1)

        outputs = model(input_ids)
        predictions = (outputs > 0.5).float()

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

acc = correct / total
print(f"✅ Accuracy sur le test set : {acc:.4f}")