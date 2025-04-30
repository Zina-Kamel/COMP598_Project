import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import TweetTokenizer
from tqdm import tqdm
from torch.nn.functional import sigmoid
import csv


class FastTextMultilabelClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_labels):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab["<pad>"])
        self.fc = nn.Linear(embed_dim, num_labels)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)              
        pooled = embedded.mean(dim=1)                    
        logits = self.fc(pooled)                          
        return logits

vocab = torch.load("fasttext_multilabel_vocab.pt")
with open("label_names.txt", "r") as f:
    label_names = f.read().splitlines()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FastTextMultilabelClassifier(vocab_size=len(vocab), embed_dim=128, num_labels=15).to(device)
model.load_state_dict(torch.load("fasttext_multilabel_model.pt"))

test_df = pd.read_csv("models/codeswitch/test_combined_separated_cs_dataset.csv")
test_df["languages"] = test_df["languages"].apply(eval)

vocab = torch.load("fasttext_multilabel_vocab.pt")
with open("label_names.txt", "r") as f:
    label_names = f.read().splitlines()

mlb = MultiLabelBinarizer(classes=label_names)
y_val = mlb.fit_transform(test_df["languages"])  
X_val = test_df["codeswitch_sentence"].tolist()

tweet_tokenizer = TweetTokenizer()

def tokenize(text):
    return tweet_tokenizer.tokenize(text.lower())

def encode(text):
    return [vocab.get(tok, vocab["<unk>"]) for tok in tokenize(text)]

class FastTextDataset(Dataset):
    def __init__(self, texts, labels, max_len=50):
        self.encoded_texts = [encode(t)[:max_len] for t in texts]
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.max_len = max_len

    def __getitem__(self, idx):
        tokens = self.encoded_texts[idx]
        pad_len = self.max_len - len(tokens)
        padded = tokens + [vocab["<pad>"]] * pad_len
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "labels": self.labels[idx]
        }

    def __len__(self):
        return len(self.labels)

val_dataset = FastTextDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=64)

model.eval()
all_preds = []
all_sentences = X_val 

with torch.no_grad():
    for batch in tqdm(val_loader, desc="Predicting"):
        input_ids = batch["input_ids"].to(device)
        outputs = model(input_ids)
        probs = sigmoid(outputs).cpu().numpy()  
        all_preds.extend(probs)

threshold = 0.8
binary_preds = (np.array(all_preds) >= threshold).astype(int)

predicted_labels = mlb.inverse_transform(binary_preds)

output_path = "fasttext_multilabel_predictions.csv"
with open(output_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["codeswitch_sentence", "predicted_languages"])
    for sentence, labels in zip(all_sentences, predicted_labels):
        writer.writerow([sentence, list(labels)])

print(f"Predictions saved to {output_path}")
