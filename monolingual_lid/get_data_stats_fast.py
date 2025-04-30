import os
import pandas as pd
import json
from collections import Counter, defaultdict
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import numpy as np
from glob import glob
import re

from itertools import islice

# Setup
os.makedirs("lid_data_stats_partial", exist_ok=True)
batch_size = 100000
# batch_idx = 0
current_batch = []

language_counter = Counter()
dataset_counter = Counter()
token_counter = Counter()
char_counter = Counter()
short_sentences_counter = Counter()
special_char_sentences = Counter()

special_characters = set("!@#$%^&*()[]{};:,./<>?\|`~-=_+\"'0123456789")
os.makedirs("lid_data_stats_partial", exist_ok=True)


# Load
# ds = load_dataset("JessicaOjo/lid-training-data", trust_remote_code=True, streaming=True)
#
#
# def process_batch(batch, batch_idx):
#     lang_counter = Counter()
#     dataset_counter = Counter()
#     token_counter = Counter()
#     char_counter = Counter()
#     short_sent_counter = Counter()
#     special_counter = Counter()
#     lengths = []
#
#     for ex in batch:
#         print(ex)
#         sentence = ex['sentence']
#         lang = ex['language']
#         dataset = ex['dataset']
#
#         tokens = sentence.split()
#         num_tokens = len(tokens)
#
#         lang_counter[lang] += 1
#         dataset_counter[dataset] += 1
#         token_counter[lang] += num_tokens
#         char_counter[lang] += len(sentence)
#         lengths.append((lang, num_tokens))
#
#         if num_tokens <= 5:
#             short_sent_counter[lang] += 1
#         if any(c in special_characters for c in sentence):
#             special_counter[lang] += 1
#
#     # Save partials to disk
#     pd.DataFrame(lengths, columns=["language", "num_tokens"]).to_parquet(f"lid_data_stats_partial/lengths_{batch_idx}.parquet")
#     pd.DataFrame(lang_counter.items(), columns=["language", "num_sentences"]).to_parquet(f"lid_data_stats_partial/lang_{batch_idx}.parquet")
#     pd.DataFrame(dataset_counter.items(), columns=["dataset", "num_sentences"]).to_parquet(f"lid_data_stats_partial/dataset_{batch_idx}.parquet")
#     pd.DataFrame(token_counter.items(), columns=["language", "total_tokens"]).to_parquet(f"lid_data_stats_partial/tokens_{batch_idx}.parquet")
#     pd.DataFrame(char_counter.items(), columns=["language", "total_chars"]).to_parquet(f"lid_data_stats_partial/chars_{batch_idx}.parquet")
#     pd.DataFrame(short_sent_counter.items(), columns=["language", "short_sentences_<=5_tokens"]).to_parquet(f"lid_data_stats_partial/short_{batch_idx}.parquet")
#     pd.DataFrame(special_counter.items(), columns=["language", "special_char_sentences"]).to_parquet(f"lid_data_stats_partial/special_{batch_idx}.parquet")
#
#
# # Stream and flush batches
# for ex in tqdm(ds, desc="Streaming examples"):
#     current_batch.append(ex)
#
#     if len(current_batch) >= batch_size:
#         process_batch(current_batch, batch_idx)
#         batch_idx += 1
#         current_batch = []
#
# # Process final partial
# if current_batch:
#     process_batch(current_batch, batch_idx)
#
# print("✅ All partials saved!")

# Load already downloaded dataset
ds = load_dataset("JessicaOjo/lid-training-data", trust_remote_code=True, download_mode="reuse_dataset_if_exists")


# Flatten if dict
def flatten_dataset(ds):
    if isinstance(ds, dict):
        return concatenate_datasets(list(ds.values()))  # Only one split in your case
    return ds


ds = flatten_dataset(ds)


def process_batch(batch, batch_idx):
    lang_counter = Counter()
    dataset_counter = Counter()
    token_counter = Counter()
    char_counter = Counter()
    short_sent_counter = Counter()
    special_counter = Counter()
    lengths = []
    for ex in batch:
        sentence = ex['sentence']
        lang = ex['language']
        dataset = ex['dataset']
        tokens = sentence.split()
        num_tokens = len(tokens)

        lang_counter[lang] += 1
        dataset_counter[dataset] += 1
        token_counter[lang] += num_tokens
        char_counter[lang] += len(sentence)
        lengths.append((lang, num_tokens))

        if num_tokens <= 5:
            short_sent_counter[lang] += 1
        if any(c in special_characters for c in sentence):
            special_counter[lang] += 1

    # Save partials to disk
    pd.DataFrame(lengths, columns=["language", "num_tokens"]).to_parquet(f"lid_data_stats_partial/lengths_{batch_idx}.parquet")
    pd.DataFrame(lang_counter.items(), columns=["language", "num_sentences"]).to_parquet(f"lid_data_stats_partial/lang_{batch_idx}.parquet")
    pd.DataFrame(dataset_counter.items(), columns=["dataset", "num_sentences"]).to_parquet(f"lid_data_stats_partial/dataset_{batch_idx}.parquet")
    pd.DataFrame(token_counter.items(), columns=["language", "total_tokens"]).to_parquet(f"lid_data_stats_partial/tokens_{batch_idx}.parquet")
    pd.DataFrame(char_counter.items(), columns=["language", "total_chars"]).to_parquet(f"lid_data_stats_partial/chars_{batch_idx}.parquet")
    pd.DataFrame(short_sent_counter.items(), columns=["language", "short_sentences_<=5_tokens"]).to_parquet(f"lid_data_stats_partial/short_{batch_idx}.parquet")
    pd.DataFrame(special_counter.items(), columns=["language", "special_char_sentences"]).to_parquet(f"lid_data_stats_partial/special_{batch_idx}.parquet")

    print(f"✅ Saved batch {batch_idx}")


def get_last_batch_idx(partial_dir="lid_data_stats_partial"):
    batch_files = os.listdir(partial_dir)
    batch_indices = []
    for fname in batch_files:
        match = re.search(r'_(\d+)\.parquet$', fname)
        if match:
            batch_indices.append(int(match.group(1)))
    if batch_indices:
        return max(batch_indices) + 1  # Start from next batch
    else:
        return 0  # No files yet


# Usage
batch_idx = 5669
print(f"🔵 Resuming from batch index {batch_idx}")
examples_to_skip = batch_size * batch_idx

# Efficiently skip examples
# ds_iter = islice(ds, examples_to_skip, None)
# ds_iter = ds[585700000:]
# print(f"🔵 Skipping {examples_to_skip} examples...")

for idx, ex in tqdm(enumerate(ds), desc="Processing examples"):
    if idx < examples_to_skip:
        continue
    # print(f"🔵 Skipped {examples_to_skip} examples...")
    current_batch.append(ex)

    if len(current_batch) >= batch_size:
        process_batch(current_batch, batch_idx)
        batch_idx += 1
        current_batch = []

# Final leftover
if current_batch:
    process_batch(current_batch, batch_idx)

print("✅ All batches processed!")


# concatenate datastats
def merge_parquets(folder, prefix):
    files = sorted(glob(os.path.join(folder, f"{prefix}_*.parquet")))
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


# Merge all partials
lengths_df = merge_parquets("lid_data_stats_partial", "lengths")
lang_df = merge_parquets("lid_data_stats_partial", "lang")
dataset_df = merge_parquets("lid_data_stats_partial", "dataset")
tokens_df = merge_parquets("lid_data_stats_partial", "tokens")
chars_df = merge_parquets("lid_data_stats_partial", "chars")
shorts_df = merge_parquets("lid_data_stats_partial", "short")
specials_df = merge_parquets("lid_data_stats_partial", "special")

print("✅ Partial files loaded and concatenated.")

# Aggregate totals per language
lang_summary = lang_df.groupby("language")["num_sentences"].sum().reset_index()
tokens_summary = tokens_df.groupby("language")["total_tokens"].sum().reset_index()
chars_summary = chars_df.groupby("language")["total_chars"].sum().reset_index()
shorts_summary = shorts_df.groupby("language")["short_sentences_<=5_tokens"].sum().reset_index()
specials_summary = specials_df.groupby("language")["special_char_sentences"].sum().reset_index()

# Sentence lengths: aggregate mean, median, min, max per language
agg_lengths = lengths_df.groupby("language")["num_tokens"].agg(["mean", "median", "min", "max"]).reset_index()

# Final join
full_lang_summary = lang_summary.merge(tokens_summary, on="language")\
                                .merge(chars_summary, on="language")\
                                .merge(shorts_summary, on="language")\
                                .merge(specials_summary, on="language")\
                                .merge(agg_lengths, on="language")

# Dataset stats
dataset_summary = dataset_df.groupby("dataset")["num_sentences"].sum().reset_index()

print("✅ Final aggregation done!")

os.makedirs("lid_data_stats_final", exist_ok=True)

full_lang_summary.to_csv("lid_data_stats_final/language_summary.csv", index=False)
dataset_summary.to_csv("lid_data_stats_final/dataset_summary.csv", index=False)

global_summary = {
    "total_sentences": int(lang_summary["num_sentences"].sum()),
    "total_languages": int(lang_summary.shape[0]),
    "total_datasets": int(dataset_summary.shape[0]),
}
with open("lid_data_stats_final/global_summary.json", "w") as f:
    json.dump(global_summary, f, indent=2)

print("✅ Stats saved!")

