import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import TweetTokenizer
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt

train_df = pd.read_csv("models/codeswitch/train_combined_separated_cs_dataset.csv")

train_df["languages"] = train_df["languages"].apply(eval)

mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(train_df["languages"])

X_train = train_df["codeswitch_sentence"].tolist()
label_names = mlb.classes_

tweet_tokenizer = TweetTokenizer()

def tokenize(text):
    return tweet_tokenizer.tokenize(text.lower())

counter = Counter()
for text in X_train:
    counter.update(tokenize(text))

vocab = {word: i+2 for i, (word, _) in enumerate(counter.most_common())}
vocab["<pad>"] = 0
vocab["<unk>"] = 1

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

train_dataset = FastTextDataset(X_train, y_train)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_labels = len(mlb.classes_)

model = FastTextMultilabelClassifier(vocab_size=len(vocab), embed_dim=128, num_labels=num_labels).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train_model(model, train_loader, epochs=5):
    train_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss:.4f}")

    return train_losses

train_losses = train_model(model, train_loader, epochs=5)

torch.save(model.state_dict(), "fasttext_multilabel_model.pt")
torch.save(vocab, "fasttext_multilabel_vocab.pt")
with open("label_names.txt", "w") as f:
    f.write("\n".join(label_names))

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Training Loss", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve_fasttext_multilabel.png") 
plt.show()
