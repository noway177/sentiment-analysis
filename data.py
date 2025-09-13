from datasets import load_dataset
import re
from collections import Counter
from collections import Counter
from torch.utils.data import Dataset,DataLoader
import torch

# On télécharge le dataset IMDb
dataset = load_dataset("imdb")

# On regarde les clés disponibles (train, test, unsupervised)
print(dataset)
#le dataset est : DatasetDict({train: Dataset({ features: ['text', 'label'],num_rows: 25000})
                #              test: Dataset({ features: ['text', 'label'], num_rows: 25000})
                #              unsupervised: Dataset({features: ['text', 'label'],num_rows: 50000})})



#fonction qui permet de nettoyer les données pour qu'elles soit plus clean 
def clean_text(text):
    text = text.lower()  # tout en minuscules (diminue la taille du vocabulaire)
    text = re.sub(r"<br\s*/?>", " ", text)  # supprime les balises <br /> (rend le texte plus comprhénesible)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # supprime la ponctuation (diminiue la taille du vocabulaire)
    return text

dataset = dataset.map(lambda x: {"text": clean_text(x["text"])}) #on nettoie les textes de notre dataset
#x est un dict du dataset :{"text": "This movie was AWESOME!<br /> Totally worth it.","label": }
#on applique clean text au text du dictionnaire 
#on créé une fonction lambda qui a x modifie text en clean text
#.map() -> focntion hugging face qui manipule les dataset et qui permet en gros d'allez dans le dataset et de bien changez le text car indiquez ici 

def tokenize(text):
    return text.split()  #on met les mots dans une liste ( une mot en liste [0], un mot en liste[1]... )



# On applique la tokenisation à tout le dataset, on créé une nouvelle colonne qui contient la liste  
tokenized_dataset = dataset.map(lambda x: {"tokens": tokenize(x["text"])})

# On construit le vocabulaire à partir des tokens du train
counter = Counter() #Class counter qui permettra ensuite de compter les occurences

examples = list(tokenized_dataset["train"]) #on fait un liste des dataset(text,label,token) de train 
for ex in examples:                         #on parcours dataset par dataset
    counter.update(ex["tokens"])            #on compte le nombre d'occurence de chaque mot et on les mets dans le counter
                                            #le counter est sous la forme suivante : Counter({"the": 10543,"movie": 9320,"this": 8761,"was": 8002,...})
# On garde les N mots les plus fréquents (par exemple 20 000)
vocab_size = 20000
most_common = counter.most_common(vocab_size - 2)  # on garde deux places pour <unk> et <pad>

# On crée les dictionnaires mot ↔ index
word2idx = {word: idx + 2 for idx, (word, _) in enumerate(most_common)} #on associe un token unique à chacun des 1998 moi les plus utilisé 
                                                                        #enumerate ca nous donne un idx pour chaque words/nbr d'apparition
                                                                        #on ignore le nbr d'apparition et on ajoute  dans le dicitonnaire wrd:idx+2
word2idx["<pad>"] = 0       #ca rajoutera des 0 à la fin des phrases trop courtes                                              
word2idx["<unk>"] = 1       #ca remplacera les mots inconnu par des 1                                       

# {  "<pad>": 0, "<unk>": 1,  "the": 2,  "movie": 3,  "was": 4,  ...} le format quon a l'heure actuelle 

idx2word = {idx: word for word, idx in word2idx.items()}


def vectorize(tokens, word2idx, max_len=100):
    # On convertit chaque token en son index dans le vocabulaire
    ids = [word2idx.get(token, word2idx["<unk>"]) for token in tokens] #on cgerche dans word2idx le token pour token dans tokens et si on le trouve pas on met 1

    # On applique du padding si la séquence est trop courte
    if len(ids) < max_len:
        ids += [word2idx["<pad>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]  # On coupe si trop long

    return ids

max_len = 100  # longueur fixe pour toutes les séquences

tokenized_dataset = tokenized_dataset.map(
    lambda x: {"input_ids": vectorize(x["tokens"], word2idx, max_len)}
)

class IMDBDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "label": torch.tensor(item["label"], dtype=torch.float)
        }



train_dataset = IMDBDataset(tokenized_dataset["train"])
test_dataset = IMDBDataset(tokenized_dataset["test"])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)