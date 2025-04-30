from collections import Counter
import os
import pandas as pd
from tqdm import tqdm

raw_path = "training_data_capped_100/fixed_capped.txt"

language_counter = Counter()
token_counter = Counter()

with open(raw_path, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Processing raw text"):
        line = line.strip()
        if not line:
            continue
        if not line.startswith("__label__"):
            continue

        try:
            label, sentence = line.split(' ', 1)
            lang = label.replace("__label__", "").strip()
            tokens = sentence.split()

            language_counter[lang] += 1
            token_counter[lang] += len(tokens)
        except ValueError:
            continue  # skip broken lines

# Assemble dataframe
df = pd.DataFrame({
    "language": list(language_counter.keys()),
    "num_sentences": [language_counter[l] for l in language_counter],
    "num_tokens": [token_counter[l] for l in language_counter],
})

df["avg_tokens_per_sentence"] = df["num_tokens"] / df["num_sentences"]

# Save
os.makedirs("lid_data_stats", exist_ok=True)
df.to_csv("lid_data_stats/capped_language_summary.csv", index=False)
print("Saved fast language stats to lid_data_stats/capped_language_summary.csv")
