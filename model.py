import torch.nn as nn

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim) #embedding -> transformé les idx en vecteurs 
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True) #À chaque mot, le LSTM : Lit le vecteur du mot actuel (ex: "great")
                                                                                                # Utilise ce qu’il a retenu jusque-là
                                                                                                # Décide :
                                                                                                # quoi retenir
                                                                                                # quoi oublier
                                                                                                # quoi ajouter à la mémoire
                                                                                                # Produit une nouvelle mémoire interne + une sortie
        self.fc = nn.Linear(hidden_dim, 1) # couche de neurone classique :  applique une transformation affine (donc linéaire) à un vecteur d’entrée.
        self.sigmoid = nn.Sigmoid() # fonction d’activation qui transforme un nombre en une valeur entre 0 et 1

    def forward(self, x):
        x = self.embedding(x)        # [batch_size, seq_len] → [batch_size, seq_len, emb_dim]
        x, _ = self.lstm(x)          # → [batch_size, seq_len, hidden_dim] #on juge mot 1 puis mot 1 mot 2 ... (un pue plus complee voir lstm)
        x = x[:, -1, :]              # on prend la dernière sortie (dernier mot) #on prend dcp le sens final qui prend en compte tt la phrase 
        x = self.fc(x)               # → [batch_size, 1]
        x = self.sigmoid(x)         # → probabilité
        return x
