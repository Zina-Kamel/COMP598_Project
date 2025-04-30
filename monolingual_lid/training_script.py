from datasets import load_dataset, Dataset, concatenate_datasets
import random
from tqdm import tqdm
import os
import fasttext
import tempfile
import pandas as pd
import numpy as np
from collections import defaultdict, Counter

from sklearn.metrics import accuracy_score, f1_score

from itertools import product
from joblib import Parallel, delayed
import os
from glob import glob

# from eval import smol_sent, smol_doc

iso_codes = pd.read_csv('langs/iso_code_mapping.csv')
iso_1_to_2 = dict(zip(iso_codes['iso_1'], iso_codes['iso_3']))

langs = pd.read_csv('langs/bloom_languages.csv')
bloom_langs = dict(zip(langs['ISO 639-3'], langs['Name']))

smol_sent = ['en_aa', 'en_ach', 'en_aeb', 'en_af', 'en_ak', 'en_alz', 'en_am', 'en_ar-MA', 'en_arz', 'en_ay',
             'en_ayl', 'en_bci', 'en_bem', 'en_ber', 'en_ber-Latn', 'en_bm', 'en_bo', 'en_brx', 'en_cgg', 'en_din',
             'en_dov', 'en_dyu', 'en_ee', 'en_efi', 'en_es', 'en_ff', 'en_fon', 'en_gaa', 'en_gn', 'en_ha', 'en_ig',
             'en_kg', 'en_ki', 'en_kl', 'en_kr', 'en_kri', 'en_ks', 'en_ks-Deva', 'en_ktu', 'en_lg', 'en_ln', 'en_lu',
             'en_luo', 'en_lus', 'en_mfe', 'en_mg', 'en_mni-Mtei', 'en_mos', 'en_nd', 'en_ndc', 'en_nr', 'en_nso',
             'en_ny', 'en_om', 'en_pa-Arab', 'en_pcm', 'en_qu', 'en_rn', 'en_rw', 'en_sa', 'en_sat', 'en_sat-Latn',
             'en_sn', 'en_so', 'en_ss', 'en_st', 'en_sus', 'en_sw', 'en_ti', 'en_tiv', 'en_tn', 'en_ts', 'en_tum',
             'en_ve', 'en_wo', 'en_xh', 'en_yo', 'en_yue', 'en_zu',]

smol_doc = ['en_ahr', 'en_bfq', 'en_bfy', 'en_bgq', 'en_bns', 'en_bra', 'en_ccp-Latn', 'en_dhd',
            'en_doi', 'en_grt-Latn', 'en_hoc-Wara', 'en_kfy', 'en_kru', 'en_lep', 'en_lif-Limb', 'en_mag', 'en_mjl',
            'en_mtr', 'en_ne', 'en_noe', 'en_scl', 'en_scn', 'en_sd-Deva', 'en_sgj', 'en_sjp', 'en_spv', 'en_tcy',
            'en_trp', 'en_unr-Deva', 'en_wbr', 'en_xnr', 'en_xsr-Tibt']

common_voice_langs = ['en', 'fa', 'fr', 'es', 'sl', 'kab', 'cy', 'ca', 'de', 'tt', 'ta', 'ru', 'nl', 'it', 'eu', 'tr',
                      'ar', 'zh-TW', 'br', 'pt', 'eo', 'zh-CN', 'id', 'ia', 'lv', 'ja', 'rw', 'sv-SE', 'cnh', 'et',
                      'ky', 'ro', 'hsb', 'el', 'cs', 'pl', 'rm-sursilv', 'rm-vallader', 'mn', 'zh-HK', 'ab', 'cv',
                      'uk', 'mt', 'as', 'ka', 'fy-NL', 'dv', 'pa-IN', 'vi', 'or', 'ga-IE', 'fi', 'hu', 'th', 'lt',
                      'lg', 'hi', 'bas', 'sk', 'kmr', 'bg', 'kk', 'ba', 'gl', 'ug', 'hy-AM', 'be', 'ur', 'gn', 'sr',
                      'uz', 'mr', 'da', 'myv', 'nn-NO', 'ha', 'ckb', 'ml', 'mdf', 'sw', 'sat', 'tig', 'ig', 'nan-tw',
                      'mhr', 'bn', 'tok', 'yue', 'sah', 'mk', 'sc', 'skr', 'ti', 'mrj', 'tw', 'ko', 'yo', 'vot', 'az',
                      'ast', 'ne-NP', 'quy', 'oc']


def stream_and_save_raw_text(hf_repo_id, output_dir="training_data"):
    os.makedirs(output_dir, exist_ok=True)
    raw_path = os.path.join(output_dir, "raw_unsorted.txt")

    ds = load_dataset(hf_repo_id, trust_remote_code=True, streaming=False, download_mode="reuse_dataset_if_exists")

    def flatten_dataset(ds):
        if isinstance(ds, dict):
            return concatenate_datasets(list(ds.values()))
        else:
            return ds

    all_examples = flatten_dataset(ds)

    # Step 1: Stream and write raw file
    with open(raw_path, "w", encoding="utf-8") as f:
        for row in tqdm(all_examples, desc="Streaming and saving raw examples"):
            sentence = str(row["sentence"]).strip()
            lang = row["language"]
            if sentence and lang:
                f.write(f"__label__{lang} {sentence}\n")

    print(f"✅ Raw file saved at {raw_path}")


def split_shuffled_file(shuffled_path, train_path, dev_path, train_split=0.9):
    total_lines = 619257860

    split_idx = int(total_lines * train_split)
    current_idx = 0

    with open(shuffled_path, "r", encoding="utf-8") as f, \
         open(train_path, "w", encoding="utf-8") as f_train, \
         open(dev_path, "w", encoding="utf-8") as f_dev:

        for line in tqdm(f, total=total_lines, desc="Splitting into train/dev"):
            if current_idx < split_idx:
                f_train.write(line)
            else:
                f_dev.write(line)
            current_idx += 1


def chunk_shuffle_split(capped_path, shuffled_path, train_path, dev_path, train_split=0.9):
    temp_dir = tempfile.mkdtemp()
    chunk_size = 100_000
    chunk_files = []
    current_chunk = []

    with open(capped_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading and chunk-shuffling"):
            current_chunk.append(line)
            if len(current_chunk) >= chunk_size:
                random.shuffle(current_chunk)
                temp_chunk_path = os.path.join(temp_dir, f"chunk_{len(chunk_files)}.txt")
                with open(temp_chunk_path, "w", encoding="utf-8") as temp_f:
                    temp_f.writelines(current_chunk)
                chunk_files.append(temp_chunk_path)
                current_chunk = []

    if current_chunk:
        random.shuffle(current_chunk)
        temp_chunk_path = os.path.join(temp_dir, f"chunk_{len(chunk_files)}.txt")
        with open(temp_chunk_path, "w", encoding="utf-8") as temp_f:
            temp_f.writelines(current_chunk)
        chunk_files.append(temp_chunk_path)

    with open(shuffled_path, "w", encoding="utf-8") as out_f:
        for chunk_file in tqdm(chunk_files, desc="Merging shuffled chunks"):
            with open(chunk_file, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    out_f.write(line)

    print("✅ Shuffled file created.")

    # Now split
    total_lines = sum(1 for _ in open(shuffled_path, encoding="utf-8"))
    split_idx = int(total_lines * train_split)

    with open(shuffled_path, "r", encoding="utf-8") as f, \
         open(train_path, "w", encoding="utf-8") as f_train, \
         open(dev_path, "w", encoding="utf-8") as f_dev:

        for i, line in tqdm(enumerate(f), total=total_lines, desc="Splitting into train/dev"):
            if i < split_idx:
                f_train.write(line)
            else:
                f_dev.write(line)

    print(f"✅ Train/dev split: {split_idx} train lines, {total_lines - split_idx} dev lines")


def prepare_fasttext_data_capped_safe(output_dir="training_data_capped"):
    os.makedirs(output_dir, exist_ok=True)
    raw_path = os.path.join("training_data", "raw_unsorted.txt")
    temp_fixed_path = os.path.join(output_dir, "temp_fixed.txt")
    # capped_path = os.path.join(output_dir, "fixed_capped.txt")
    # shuffled_path = os.path.join(output_dir, "shuffled.txt")
    # train_path = os.path.join(output_dir, "train.txt")
    # dev_path = os.path.join(output_dir, "dev.txt")

    # --- Step 1: Fix ISO codes ---
    with open(raw_path, "r", encoding="utf-8") as f_in, open(temp_fixed_path, "w", encoding="utf-8") as f_out:
        for line in tqdm(f_in, desc="Fixing ISO codes"):
            if not line.strip() or not line.startswith("__label__"):
                continue
            try:
                label, sentence = line.split(" ", 1)
                lang = label.replace("__label__", "")
                if len(lang) == 2 and lang in iso_1_to_2:
                    lang = iso_1_to_2[lang]
                f_out.write(f"__label__{lang} {sentence}")
            except Exception:
                continue

    print(f"✅ Temp fixed file saved at {temp_fixed_path}")


def shuffle_and_split(output_dir="training_data_capped_100", train_split=0.9):
    os.makedirs(output_dir, exist_ok=True)
    temp_fixed_path = os.path.join("training_data_capped", "temp_fixed.txt")
    capped_path = os.path.join(output_dir, "fixed_capped.txt")
    shuffled_path = os.path.join(output_dir, "shuffled.txt")
    train_path = os.path.join(output_dir, "train.txt")
    dev_path = os.path.join(output_dir, "dev.txt")
    print("🔵 Chunked shuffle of temp_fixed file...")

    temp_dir = tempfile.mkdtemp()
    chunk_size = 100_000
    chunk_files = []
    current_chunk = []

    # 1. Split into shuffled chunks
    with open(temp_fixed_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading temp_fixed.txt"):
            current_chunk.append(line)
            if len(current_chunk) >= chunk_size:
                random.shuffle(current_chunk)
                temp_chunk_path = os.path.join(temp_dir, f"chunk_{len(chunk_files)}.txt")
                with open(temp_chunk_path, "w", encoding="utf-8") as temp_f:
                    temp_f.writelines(current_chunk)
                chunk_files.append(temp_chunk_path)
                current_chunk = []

    if current_chunk:
        random.shuffle(current_chunk)
        temp_chunk_path = os.path.join(temp_dir, f"chunk_{len(chunk_files)}.txt")
        with open(temp_chunk_path, "w", encoding="utf-8") as temp_f:
            temp_f.writelines(current_chunk)
        chunk_files.append(temp_chunk_path)

    print("✅ Chunked shuffle complete.")

    print("🔵 Counting examples per language...")
    lang_total_counter = Counter()

    for chunk_file in tqdm(chunk_files, desc="Counting languages"):
        with open(chunk_file, "r", encoding="utf-8") as in_f:
            for line in in_f:
                if not line.strip():
                    continue
                label, _ = line.split(" ", 1)
                lang = label.replace("__label__", "")
                lang_total_counter[lang] += 1

    # Second pass: Write only langs >= 100 examples, and cap at 1M
    print("🔵 Capping and writing filtered languages...")
    lang_written_counter = Counter()

    with open(capped_path, "w", encoding="utf-8") as f_out:
        for chunk_file in tqdm(chunk_files, desc="Processing shuffled chunks"):
            with open(chunk_file, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    if not line.strip():
                        continue
                    label, _ = line.split(" ", 1)
                    lang = label.replace("__label__", "")
                    if lang_total_counter[lang] >= 100 and lang_written_counter[lang] < 1_000_000:
                        f_out.write(line)
                        lang_written_counter[lang] += 1

    print(f"✅ Capped file saved at {capped_path}")

    chunk_shuffle_split(capped_path, shuffled_path, train_path, dev_path, train_split)


def train_fasttext_with_blocks(train_file, dev_file, output_dir="fasttext_model_tuned2",
                               total_epochs=30, block_size=10, lr=0.1, dim=200, wordNgrams=3, loss="hs",
                               bucket=1_000_000, thread=68,):
    os.makedirs(output_dir, exist_ok=True)
    model_name = f"model_lr{lr}_dim{dim}_ngram{wordNgrams}_loss{loss}.bin"
    model_path = os.path.join(output_dir, model_name)
    # model_path = os.path.join(output_dir, "model.bin")

    epochs_done = 0
    best_dev_accuracy = 0

    while epochs_done < total_epochs:
        print(f"🔵 Training block: epochs {epochs_done + 1} to {epochs_done + block_size}")

        model = fasttext.train_supervised(
            input=train_file,
            epoch=block_size,
            lr=lr,
            dim=dim,
            wordNgrams=wordNgrams,
            loss=loss,
            bucket=bucket,
            thread=thread,
            verbose=2,
        )

        model.save_model(model_path)
        print(f"✅ Saved model after {epochs_done + block_size} epochs at {model_path}")

        if dev_file:
            print("🔵 Evaluating on dev set...")
            result = model.test(dev_file)
            dev_accuracy = result[1] * 100
            print(f"📈 Dev accuracy: {dev_accuracy:.2f}% ({result[0]} samples)")

            # Early stopping condition
            if dev_accuracy <= best_dev_accuracy:
                print("🛑 Dev accuracy did not improve. Stopping training.")
                break

            best_dev_accuracy = dev_accuracy

        epochs_done += block_size

    print(f"🏁 Training complete. Best dev accuracy: {best_dev_accuracy:.2f}%")

    print("⚡ Running final evaluations on testsets...")
    flores_acc, smol_acc, smol_f1, flores_f1 = evaluate_testsets(model_path, model_name)

    return best_dev_accuracy, flores_acc, smol_acc, smol_f1, flores_f1


def evaluate_testsets(model_path, model_name):
    print("⚡ Running final evaluations on testsets...")
    model = fasttext.load_model(model_path)
    output_dir = "../final_model_results"
    print("Running Smol")
    smol_acc, smol_f1 = run_smols(model, model_name, output_dir)
    print("Running Tweetlid")
    tweetlid_acc, tweetlid_f1 = run_tweetlid_mono(model, model_name, output_dir)
    print("Running Bloom")
    bloom_acc, bloom_f1 = run_bloom_stories(model, model_name, output_dir)
    print("Running CV")
    cv_acc, cv_f1 = run_common_voice_stream(model, model_name, output_dir)
    print("Running Flores")
    flores_acc, flores_f1 = run_flores(model, model_name, output_dir)
    print(f"Test accuracy scores for {model_name}; "
          f"\nFlores; acc={flores_acc}, f1={flores_f1}"
          f"\nSmol; acc={smol_acc}, f1={smol_f1}"
          f"\nTweetLID; acc={tweetlid_acc}, f1={tweetlid_f1}"
          f"\nBloom; acc={bloom_acc}, f1={bloom_f1}"
          f"\nCommon Voice; acc={cv_acc}, f1={cv_f1}")
    return flores_acc, smol_acc, smol_f1, flores_f1


def model_predict(model, texts):
    predictions = [model.predict(text.replace('\n', ''), k=3) for text in texts]
    pred_prob = [np.asarray(pred[1])[0] for pred in predictions]
    pred_lang = [pred[0][0].replace("__label__", "") for pred in predictions]
    return pred_lang, pred_prob,


def calculate_accuracy(items) -> float:
    """Compute accuracy score for binary or multiclass classification."""
    golds, preds = zip(*items)
    return accuracy_score(golds, preds)


def calculate_f1(items, zero_division=0) -> float:
    """Compute F1 score for binary or multiclass classification."""
    golds, preds = zip(*items)
    return f1_score(golds, preds, average="macro", zero_division=zero_division)


def calculate_fpr(items):
    golds, preds = zip(*items)
    fp = 0
    tn = 0
    if golds[0] != preds[0]:
        fp = 1  # False positive (Predicted a different language)
    else:
        tn = 1  # True negative (Predicted the correct language)

    # Calculate FPR for this row
    if fp + tn > 0:
        fpr = fp / (fp + tn)
    else:
        fpr = 0
    return fpr


def run_flores(model, model_name, output_dir="model_results/"):
    data_name = "openlanguagedata/flores_plus"
    data = load_dataset(data_name, download_mode="reuse_dataset_if_exists")["devtest"]
    pred_lang, pred_prob = model_predict(model, data["text"])
    results = pd.DataFrame({
        "domain": data["domain"],
        "topic": data["topic"],
        "text": data["text"],
        "language": data["iso_639_3"],
        "lang_script": data["iso_15924"],
        "pred_lang": pred_lang,
        "pred_prob": pred_prob
    })
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{model_name}_flores.csv")
    results = calculate_metrics(results, 'language', 'flores', model_name)
    results.to_csv(csv_path, index=False)
    return results['acc'].mean(), results['f1'].mean()


def run_smols(model, model_name, output_dir="model_results/"):
    data_name = "google/smol"
    result_list = []

    languages = smol_sent + smol_doc
    for lang in languages:
        try:
            data = load_dataset(data_name, f"smolsent__{lang}")["train"]
            language_text = data['trg']
            domain = 'sentence'
        except ValueError:
            data = load_dataset(data_name, f"smoldoc__{lang}")["train"]
            language_text = [" ".join(text) for text in data['trgs']]
            domain = 'document'

        clean_texts = [text.replace("\n", " ").strip() for text in language_text]
        pred_lang, pred_prob = model_predict(model, clean_texts)
        results = pd.DataFrame({
            "text": language_text,
            "language": data["tl"],
            "pred_lang": pred_lang,
            "pred_prob": pred_prob,
            "domain": domain
        })
        result_list.append(results)

    result_df = pd.concat(result_list, ignore_index=True) if result_list else pd.DataFrame()
    result_df = calculate_metrics(result_df, 'language', 'smol', model_name)
    csv_path = os.path.join(output_dir, f"{model_name}_smol.csv")
    result_df.to_csv(csv_path, index=False)
    return result_df['acc'].mean(), result_df['f1'].mean()


def run_bloom_stories(model, model_name, output_dir="results/"):
    data_name = "sil-ai/bloom-lm"
    result_list = []
    langauges = bloom_langs.keys()
    for language in langauges:
        data = load_dataset(data_name, language)["test"]
        pred_lang, pred_prob = model_predict(model, data["text"])
        results = pd.DataFrame({
            "text": data["text"],
            "title": data['title'],
            "language_code": language,
            "language": bloom_langs[language],
            "pred_lang": pred_lang,
            "pred_prob": pred_prob,
            "license": data['license'],
            "copyright": data['copyright'],
            "pageCount": data['pageCount'],
            "bookInstanceId": data['bookInstanceId'],
            "bookLineage": data['bookLineage']
        })
        result_list.append(results)

    os.makedirs(output_dir, exist_ok=True)
    result_df = pd.concat(result_list, ignore_index=True) if result_list else pd.DataFrame()
    result_df = calculate_metrics(result_df, 'language_code', 'bloom_stories', model_name)
    csv_path = os.path.join(output_dir, f"{model_name}_bloom_stories.csv")
    result_df.to_csv(csv_path, index=False)
    return result_df['acc'].mean(), result_df['f1'].mean()


def run_tweetlid_mono(model, model_name, output_dir="results/tweetlid/"):
    data = pd.read_csv('../dataset/tweetlid_test_mono.csv')
    pred_lang, pred_prob = model_predict(model, data["tweet"])
    data['pred_lang'] = pred_lang
    data['pred_prob'] = pred_prob

    os.makedirs(output_dir, exist_ok=True)
    data = calculate_metrics(data, 'lang_code', 'tweetlid', model_name)
    csv_path = os.path.join(output_dir, f"{model_name}_tweetlid.csv")
    data.to_csv(csv_path, index=False)
    return data['acc'].mean(), data['f1'].mean()


def run_common_voice(model, model_name, output_dir="results/common_voice/"):
    data_name = "mozilla-foundation/common_voice_12_0"
    result_list = []
    langauges = common_voice_langs
    for language in langauges:
        data = load_dataset(data_name, language, split="test", trust_remote_code=True, download_mode="reuse_dataset_if_exists")
        pred_lang, pred_prob = model_predict(model, data["sentence"])
        results = pd.DataFrame({
            "text": data["sentence"],
            "language_code": language,
            "pred_lang": pred_lang,
            "pred_prob": pred_prob,
            "age": data['age'],
            "gender": data['gender'],
            "accent": data['accent'],
            "up_votes": data['up_votes'],
            "down_votes": data['down_votes']
        })
        result_list.append(results)

    os.makedirs(output_dir, exist_ok=True)
    result_df = pd.concat(result_list, ignore_index=True) if result_list else pd.DataFrame()
    result_df = calculate_metrics(result_df, 'language_code', 'common_voice', model_name)
    csv_path = os.path.join(output_dir, f"{model_name}_common_voice.csv")
    result_df.to_csv(csv_path, index=False)
    return result_df['acc'].mean(), result_df['f1'].mean()


def run_common_voice_stream(model, model_name, output_dir="results/common_voice/"):
    data_name = "mozilla-foundation/common_voice_12_0"
    result_list = []
    languages = common_voice_langs

    for language in languages:
        data = load_dataset(data_name, language, split="test", streaming=True, trust_remote_code=True)

        texts, ages, genders, accents, up_votes, down_votes = [], [], [], [], [], []

        for example in data:
            texts.append(example.get("sentence", ""))
            ages.append(example.get("age", ""))
            genders.append(example.get("gender", ""))
            accents.append(example.get("accent", ""))
            up_votes.append(example.get("up_votes", 0))
            down_votes.append(example.get("down_votes", 0))

        pred_lang, pred_prob = model_predict(model, texts)

        results = pd.DataFrame({
            "text": texts,
            "language_code": language,
            "pred_lang": pred_lang,
            "pred_prob": pred_prob,
            "age": ages,
            "gender": genders,
            "accent": accents,
            "up_votes": up_votes,
            "down_votes": down_votes
        })
        result_list.append(results)
    os.makedirs(output_dir, exist_ok=True)
    result_df = pd.concat(result_list, ignore_index=True) if result_list else pd.DataFrame()
    result_df = calculate_metrics(result_df, 'language_code', 'common_voice', model_name)
    csv_path = os.path.join(output_dir, f"{model_name}_common_voice.csv")
    result_df.to_csv(csv_path, index=False)
    return result_df['acc'].mean(), result_df['f1'].mean()


def calculate_metrics(df, lang_column, data_name, model_name):
    df.dropna(subset=[lang_column, 'pred_lang'], inplace=True)
    df = df[(df[lang_column] != 'und') & (df[lang_column] != 'other')]
    df['pred_name'] = df.apply(
                            lambda row: iso_1_to_2.get(row['pred_lang'].split('_')[0].split('-')[0], row['pred_lang'])
                            if len(row['pred_lang']) == 2 else row['pred_lang'],
                            axis=1
                        )
    df['gold_name'] = df.apply(
                            lambda row: iso_1_to_2.get(row[lang_column].split('_')[0].split('-')[0], row[lang_column])
                            if len(row[lang_column]) == 2 else row[lang_column],
                            axis=1
                        )

    df['acc'] = df.apply(lambda row: calculate_accuracy([(row['gold_name'], row['pred_name'])]), axis=1)
    df['f1'] = df.apply(lambda row: calculate_f1([(row['gold_name'], row['pred_name'])]), axis=1)
    df['fpr'] = df.apply(lambda row: calculate_fpr([(row['gold_name'], row['pred_name'])]), axis=1)

    mean_metrics = {
            'data_name': data_name,
            'model': model_name,
            'acc': round(df['acc'].mean()*100, 4),
            'f1': round(df['f1'].mean()*100, 4),
            'fpr': round(df['fpr'].mean()*100, 4),
        }
    results_df = pd.DataFrame([mean_metrics])
    result_path = f'model_results/result_summary.csv'
    if os.path.exists(result_path):
        results_df.to_csv(result_path, mode='a', header=False, index=False)
    else:
        results_df.to_csv(result_path, mode='w', header=True, index=False)
    return df


def train_one_config(train_file, dev_file, output_base_dir, config, total_epochs, block_size, bucket, thread):
    lr, dim, wordNgrams, loss = config

    output_dir = os.path.join(
        output_base_dir,
        f"model_lr{lr}_dim{dim}_ngram{wordNgrams}_loss{loss}"
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"🚀 Training: lr={lr}, dim={dim}, ngrams={wordNgrams}, loss={loss}")

    dev_acc, flores_acc, smol_acc, smol_f1, flores_f1 = train_fasttext_with_blocks(
        train_file=train_file,
        dev_file=dev_file,
        output_dir=output_dir,
        total_epochs=total_epochs,
        block_size=block_size,
        lr=lr,
        dim=dim,
        wordNgrams=wordNgrams,
        loss=loss,
        bucket=bucket,
        thread=thread,
    )

    return {
        "output_dir": output_dir,
        "lr": lr,
        "dim": dim,
        "wordNgrams": wordNgrams,
        "loss": loss,
        "bucket": bucket,
        "total_epochs": total_epochs,
        "dev_accuracy": dev_acc,
        "smol_accuracy": smol_acc,
    }


def sweep_fasttext_hyperparameters_parallel(
    train_file,
    dev_file,
    output_base_dir,
    results_csv="fasttext_experiments.csv",
    lr_list=[0.05, 0.1],
    dim_list=[100, 200],
    wordNgrams_list=[2, 3],
    loss_list=["hs", "softmax"],
    total_epochs=20,
    block_size=5,
    bucket=1_000_000,
    thread=16,          # Thread per run (you can reduce depending on your CPU)
    n_jobs=4,           # Number of sweeps in parallel
):
    os.makedirs(output_base_dir, exist_ok=True)

    sweep_configs = list(product(lr_list, dim_list, wordNgrams_list, loss_list))
    print(f"🔵 Starting parallel sweep: {len(sweep_configs)} configurations (n_jobs={n_jobs})")

    results = Parallel(n_jobs=n_jobs)(
        delayed(train_one_config)(
            train_file, dev_file, output_base_dir, config, total_epochs, block_size, bucket, thread
        )
        for config in sweep_configs
    )

    # Save final sweep results
    df = pd.DataFrame(results)
    df.to_csv(results_csv, index=False)
    print(f"\n✅ Full hyperparameter sweep saved to {results_csv}")


def sweep_results(model_dir):
    # model_paths = glob(f"{model_dir}/model.bin")
    # model_paths.append('fasttext_sweep_models/model_lr0.1_dim200_ngram2_losssoftmax/model.bin')
    # model_paths.append('fasttext_sweep_models/model_lr0.1_dim200_ngram3_losssoftmax/model.bin')
    model_paths = ['fasttext_sweep_models/model_lr0.1_dim200_ngram2_losssoftmax/model.bin']
    for model_path in model_paths:
        model_name = model_path.split('/')[-2]
        evaluate_testsets(model_path, model_name)





# shuffle_and_split()
# print("Split and train successful")

# def train_fasttext_with_blocks(train_file, dev_file=None, output_dir="fasttext_model",
#                                total_epochs=50, block_size=5, lr=0.05, dim=100, wordNgrams=3, loss="hs", thread=64)
# train_fasttext_with_blocks("training_data_capped_100/train.txt", "training_data_capped_100/dev.txt",
#                         total_epochs=20, block_size=5, lr=0.1, dim=300, wordNgrams=2, loss="softmax", thread=68,
#                            bucket=1_000_000,)


evaluate_testsets('fasttext_sweep_models/model_lr0.1_dim200_ngram3_losssoftmax/model.bin',
                  'model_lr0.1_dim200_ngram3_losssoftmax_smol')


# sweep_fasttext_hyperparameters_parallel(
#     train_file="training_data_capped_100/train.txt",
#     dev_file="training_data_capped_100/dev.txt",
#     output_base_dir="fasttext_sweep_models",
#     results_csv="fasttext_sweep_results.csv",
#     lr_list=[0.05, 0.1, 0.2],
#     dim_list=[100, 200],
#     wordNgrams_list=[2, 3],
#     loss_list=["hs", "softmax"],
#     total_epochs=20,
#     block_size=5,
#     n_jobs=4,
# )

# sweep_results('fasttext_sweep_models/model_lr0.2*')
