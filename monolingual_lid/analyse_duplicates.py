import os
import json
from collections import Counter
from tqdm import tqdm


def analyze_duplicates(raw_path, save_dir="lid_data_stats"):
    seen_sentences = set()
    total_lines = 0
    duplicate_count = 0

    with open(raw_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Checking duplicates"):
            line = line.strip()
            if not line:
                continue
            try:
                _, sentence = line.split(' ', 1)
            except ValueError:
                continue  # Skip broken lines

            total_lines += 1
            if sentence in seen_sentences:
                duplicate_count += 1
            else:
                seen_sentences.add(sentence)

    stats = {
        "total_sentences": total_lines,
        "unique_sentences": len(seen_sentences),
        "duplicates": duplicate_count,
        "duplicates_percentage": (duplicate_count / total_lines) * 100
    }

    print(f"Total sentences: {stats['total_sentences']}")
    print(f"Unique sentences: {stats['unique_sentences']}")
    print(f"Duplicate sentences: {stats['duplicates']} ({stats['duplicates_percentage']:.2f}%)")

    # Save
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "duplicate_stats.json")
    with open(save_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Duplicate statistics saved to {save_path}")
    return stats

analyze_duplicates("training_data/raw_unsorted.txt")
