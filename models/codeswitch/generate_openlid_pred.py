import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.tokenize import TweetTokenizer
from torch.nn.functional import sigmoid
from tqdm import tqdm
import csv

def load_embeddings(embedding_path):
    embeddings = {}
    with open(embedding_path, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.rstrip().split()
            word = tokens[0]
            vector = torch.tensor([float(v) for v in tokens[1:]], dtype=torch.float)
            embeddings[word] = vector
    return embeddings

embedding_path = "lid201-vec.txt"
openlid_embeddings = load_embeddings(embedding_path)

vocab = torch.load("openlid__finetuned_vocab.pt")

embedding_dim = len(next(iter(openlid_embeddings.values())))
embedding_matrix = torch.zeros(len(vocab), embedding_dim)
for word, idx in vocab.items():
    embedding_matrix[idx] = openlid_embeddings[word]

label_mapping = {
    'Arabizi': 'arb_Latn', 'Basque': 'eus_Latn', 'Catalan': 'cat_Latn',
    'Chinese': 'zho_Hans', 'Egyptian': 'arz_Arab', 'English': 'eng_Latn',
    'German': 'deu_Latn', 'Hindi': 'hin_Deva', 'Indonesian': 'ind_Latn',
    'MSA': 'arb_Arab', 'Malayalam': 'mal_Mlym', 'Saudi': 'ars_Arab',
    'Spanish': 'spa_Latn', 'Tamil': 'tam_Taml', 'Turkish': 'tur_Latn'
}

inv_label_mapping = {v: k for k, v in label_mapping.items()}

test_df = pd.read_csv("models/codeswitch/test_combined_separated_cs_dataset.csv")
test_df["languages"] = test_df["languages"].apply(eval).apply(lambda labels: [label_mapping[l] for l in labels])
X_val = test_df["codeswitch_sentence"].tolist()

mlb = MultiLabelBinarizer(classes=sorted(label_mapping.values()))
y_val = mlb.fit_transform(test_df["languages"])

class FastTextMultilabelClassifier(nn.Module):
    def __init__(self, embedding_matrix, num_labels):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        embedding_dim = embedding_matrix.size(1)

        self.adapter = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),  
            nn.LayerNorm(embedding_dim)
        )
        self.fc = nn.Linear(embedding_matrix.size(1), num_labels)

    def forward(self, input_ids, lengths):
        embedded = self.embedding(input_ids)
        adapted = self.adapter(embedded) 
        mask = (input_ids != 0).unsqueeze(-1)
        summed = (adapted * mask).sum(1)
        averaged = summed / lengths.unsqueeze(1).clamp(min=1)
        return self.fc(averaged)

tweet_tokenizer = TweetTokenizer()

def tokenize(text):
    return tweet_tokenizer.tokenize(text.lower())

def encode(text):
    return [vocab.get(tok, 0) for tok in tokenize(text)]

class FastTextDataset(Dataset):
    def __init__(self, texts, labels, max_len=50):
        self.encoded_texts = [torch.tensor(encode(t)[:max_len], dtype=torch.long) for t in texts]
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __getitem__(self, idx):
        return self.encoded_texts[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)

def collate_fn(batch):
    input_ids, labels = zip(*batch)
    lengths = [len(x) for x in input_ids]
    padded = nn.utils.rnn.pad_sequence(input_ids, batch_first=True)
    return {
        "input_ids": padded,
        "lengths": torch.tensor(lengths),
        "labels": torch.stack(labels)
    }

val_dataset = FastTextDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=64, collate_fn=collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FastTextMultilabelClassifier(embedding_matrix=embedding_matrix, num_labels=len(mlb.classes_)).to(device)
model.load_state_dict(torch.load("openlid_finetuned_model.pt", map_location=device))
model.eval()

all_preds = []
with torch.no_grad():
    for batch in tqdm(val_loader, desc="Predicting"):
        input_ids = batch["input_ids"].to(device)
        lengths = batch["lengths"].to(device)
        outputs = model(input_ids, lengths)
        probs = sigmoid(outputs).cpu().numpy()
        all_preds.extend(probs)

all_preds_np = np.array(all_preds)
threshold = 0.8
top_k = 2

binary_preds = np.zeros_like(all_preds_np, dtype=int)

for i, probs in enumerate(all_preds_np):
    topk_indices = np.argsort(-probs)[:top_k]
    for idx in topk_indices:
        if probs[idx] >= threshold:
            binary_preds[i, idx] = 1  

predicted_labels = mlb.inverse_transform(binary_preds)

output_path = "models/codeswitch/openlid_finetuned_predictions.csv"
with open(output_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["codeswitch_sentence", "predicted_languages"])
    for sentence, labels in zip(X_val, predicted_labels):
        writer.writerow([sentence, list(labels)])

print(f"Predictions saved to {output_path}")
