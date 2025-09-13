import torch
import re
import sys
from model import SentimentLSTM
from data import word2idx  # tu dois l’avoir conservé dans data.py

# ---------------------------
# Prétraitement
# ---------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text

def tokenize(text):
    return text.split()

def vectorize(tokens, word2idx, max_len=100):
    ids = [word2idx.get(token, word2idx["<unk>"]) for token in tokens]
    if len(ids) < max_len:
        ids += [word2idx["<pad>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids

# ---------------------------
# Modèle
# ---------------------------
vocab_size = 20000
embedding_dim = 100
hidden_dim = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SentimentLSTM(vocab_size, embedding_dim, hidden_dim)
model.load_state_dict(torch.load("sentiment_model.pth", map_location=device))
model.to(device)
model.eval()

# ---------------------------
# Entrée utilisateur
# ---------------------------
if len(sys.argv) < 2:
    print("❌ Usage : python predict.py 'votre phrase'")
    sys.exit()

text = sys.argv[1]
tokens = tokenize(clean_text(text))
ids = vectorize(tokens, word2idx)
input_tensor = torch.tensor([ids], dtype=torch.long).to(device)

# ---------------------------
# Prédiction
# ---------------------------
with torch.no_grad():
    output = model(input_tensor)
    prob = output.item()

print(f"✅ Probabilité que la critique soit positive : {prob:.4f}")
