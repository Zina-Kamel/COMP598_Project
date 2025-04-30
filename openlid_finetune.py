import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.tokenize import TweetTokenizer

label_mapping = {
    'Arabizi': 'arb_Latn', 'Basque': 'eus_Latn', 'Catalan': 'cat_Latn',
    'Chinese': 'zho_Hans', 'Egyptian': 'arz_Arab', 'English': 'eng_Latn',
    'German': 'deu_Latn', 'Hindi': 'hin_Deva', 'Indonesian': 'ind_Latn',
    'MSA': 'arb_Arab', 'Malayalam': 'mal_Mlym', 'Saudi': 'ars_Arab',
    'Spanish': 'spa_Latn', 'Tamil': 'tam_Taml', 'Turkish': 'tur_Latn'
}

train_df = pd.read_csv("models/codeswitch/train_combined_separated_cs_dataset.csv")
train_df["text"] = train_df["languages"]
train_df["languages"] = train_df["languages"].apply(eval).apply(lambda labels: [label_mapping[l] for l in labels])

mlb = MultiLabelBinarizer()
mlb.fit(train_df["languages"].tolist())

#Load OpenLID Embeddings
def load_embeddings(embedding_path):
    embeddings = {}
    with open(embedding_path, 'r', encoding='utf-8') as f:
        next(f)  # skip header
        for line in f:
            tokens = line.rstrip().split()
            word = tokens[0]
            vector = torch.tensor([float(v) for v in tokens[1:]], dtype=torch.float)
            embeddings[word] = vector
    return embeddings

embedding_path = "lid201-vec.txt"
openlid_embeddings = load_embeddings(embedding_path)
print("Loaded OpenLID embeddings")

vocab = {word: idx for idx, word in enumerate(openlid_embeddings.keys())}
embedding_dim = len(next(iter(openlid_embeddings.values())))
embedding_matrix = torch.zeros(len(vocab), embedding_dim)
for word, idx in vocab.items():
    embedding_matrix[idx] = openlid_embeddings[word]

print("created vocab")

embedding_layer = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)

tokenizer = TweetTokenizer()

class CodeSwitchDataset(Dataset):
    def __init__(self, df):
        self.texts = df["text"].tolist()
        self.labels = mlb.transform(df["languages"].tolist())

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = tokenizer.tokenize(self.texts[idx].lower())
        indices = [vocab[token] for token in tokens if token in vocab]
        labels = torch.tensor(self.labels[idx], dtype=torch.float)
        return torch.tensor(indices, dtype=torch.long), labels

def collate_fn(batch):
    texts, labels = zip(*batch)
    lengths = [len(x) for x in texts]
    padded_texts = nn.utils.rnn.pad_sequence(texts, batch_first=True)
    return padded_texts, torch.stack(labels), torch.tensor(lengths)

train_dataset = CodeSwitchDataset(train_df)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

class FastTextMultilabel(nn.Module):
    def __init__(self, embedding_layer, num_labels):
        super(FastTextMultilabel, self).__init__()
        self.embedding = embedding_layer
        self.adapter = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim)
        )

        self.fc = nn.Linear(embedding_layer.embedding_dim, num_labels)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        adapted = self.adapter(embedded)
        mask = (x != 0).unsqueeze(-1)
        summed = (adapted * mask).sum(1)
        averaged = summed / lengths.unsqueeze(1).clamp(min=1)
        return self.fc(averaged)

num_labels = len(mlb.classes_)
model = FastTextMultilabel(embedding_layer, num_labels)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for texts, labels, lengths in train_loader:
        optimizer.zero_grad()
        outputs = model(texts, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}")

torch.save(model.state_dict(), "openlid_finetuned_model.pt")
torch.save(vocab, "openlid__finetuned_vocab.pt")
