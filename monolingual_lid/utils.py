from itertools import product
from datasets import load_dataset
import pycountry
import pandas as pd
import logging
import json
import os
import random
import time
from collections import Counter
from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MANIFEST_FILE = "completed_parquet_manifest.json"
OUTPUT_DIR = "hf_shards"
HF_REPO = "JessicaOjo/lid-training-data"


tatoeba_langs = langs = [
    "ab", "acm", "ady", "af", "afb", "afh", "aii", "ain", "ajp", "akl", "aln", "am", "an", "ang", "aoz", "apc", "ar",
    "arq", "ary", "arz", "as", "ast", "avk", "awa", "ayl", "az", "ba", "bal", "bar", "be", "ber", "bg", "bho", "bjn",
    "bm", "bn", "bo", "br", "brx", "bs", "bua", "bvy", "bzt", "ca", "cay", "cbk", "ce", "ceb", "ch", "chg", "chn",
    "cho", "chr", "cjy", "ckb", "ckt", "cmn", "co", "code", "cpi", "crh", "crk", "cs", "csb", "cv", "cy", "da", "de",
    "dng", "drt", "dsb", "dtp", "dv", "dws", "ee", "egl", "el", "emx", "en", "enm", "eo", "es", "et", "eu", "ext",
    "fi", "fj", "fkv", "fo", "fr", "frm", "fro", "frr", "fuc", "fur", "fuv", "fy", "ga", "gag", "gan", "gbm", "gcf",
    "gd", "gil", "gl", "gn", "gom", "gos", "got", "grc", "gsw", "gu", "gv", "ha", "hak", "haw", "hbo", "he", "hi",
    "hif", "hil", "hnj", "hoc", "hr", "hrx", "hsb", "hsn", "ht", "hu", "hy", "ia", "iba", "id", "ie", "ig", "ii",
    "ike", "ilo", "io", "is", "it", "izh", "ja", "jam", "jbo", "jdt", "jpa", "jv", "ka", "kaa", "kab", "kam", "kek",
    "kha", "kjh", "kk", "kl", "km", "kmr", "kn", "ko", "koi", "kpv", "krc", "krl", "ksh", "ku", "kum", "kw", "kxi",
    "ky", "kzj", "la", "laa", "lad", "lb", "ldn", "lfn", "lg", "lij", "liv", "lkt", "lld", "lmo", "ln", "lo", "lt",
    "ltg", "lut", "lv", "lzh", "lzz", "mad", "mai", "max", "mdf", "mfe", "mg", "mgm", "mh", "mhr", "mi", "mic", "min",
    "mk", "ml", "mn", "mni", "mnw", "moh", "mr", "mt", "mvv", "mwl", "mww", "my", "myv", "na", "nah", "nan", "nb",
    "nch", "nds", "ngt", "ngu", "niu", "nl", "nlv", "nn", "nog", "non", "nov", "npi", "nst", "nus", "nv", "ny",
    "nys", "oar", "oc", "ofs", "ood", "or", "orv", "os", "osp", "ota", "otk", "pa", "pag", "pal", "pam", "pap",
    "pau", "pcd", "pdc", "pes", "phn", "pi", "pl", "pms", "pnb", "ppl", "prg", "ps", "pt", "qu", "quc", "qya", "rap",
    "rif", "rm", "rn", "ro", "rom", "ru", "rue", "rw", "sa", "sah", "sc", "scn", "sco", "sd", "sdh", "se", "sg",
    "sgs", "shs", "shy", "si", "sjn", "sl", "sm", "sma", "sn", "so", "sq", "sr", "stq", "su", "sux", "sv", "swg",
    "swh", "syc", "ta", "te", "tet", "tg", "th", "thv", "ti", "tig", "tk", "tl", "tlh", "tly", "tmr", "tmw", "tn",
    "to", "toi", "tok", "tpi", "tpw", "tr", "ts", "tt", "tts", "tvl", "ty", "tyv", "tzl", "udm", "ug", "uk", "umb",
    "ur", "uz", "vec", "vep", "vi", "vo", "vro", "wa", "war", "wo", "wuu", "xal", "xh", "xqa", "yi", "yo", "yue",
    "zlm", "zsm", "zu", "zza"
]


def check_tatoeba_lang_pairs():
    valid_pairs = []

    for lang1, lang2 in product(tatoeba_langs, repeat=2):
        if lang1 == lang2:
            continue
        try:
            ds = load_dataset("Helsinki-NLP/tatoeba", lang1=lang1, lang2=lang2, split="train", streaming=True)
            sample = next(iter(ds), None)
            if sample:
                print(f"{lang1}-{lang2}")
                valid_pairs.append({"lang1": lang1, "lang2": lang2})
        except Exception as e:
            xx = e

    return valid_pairs


def iso_mapping():
    rows = []
    for lang in pycountry.languages:
        if hasattr(lang, 'alpha_2'):
            iso1 = lang.alpha_2
            iso2b = getattr(lang, 'bibliographic', None)
            iso2t = getattr(lang, 'terminology', None)
            iso3 = lang.alpha_3
            name = lang.name
            rows.append({
                "iso_1": iso1,
                "iso_2b": iso2b,
                "iso_2t": iso2t,
                "iso_3": iso3,
                "name": name
            })

    df_iso_mapping = pd.DataFrame(rows)
    df_iso_mapping.to_csv('langs/iso_code_mapping.csv', index=False)


def safe_map(ds, func, remove_columns):
    try:
        return ds.map(func, remove_columns=remove_columns)
    except Exception as e:
        logger.warning(f"Mapping failed: {e}")
        return ds


def safe_load(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        logger.warning(f"Loading failed for {fn.__name__} with args={args}: {e}")
        return None


def flatten_if_needed(ds):
    try:
        return ds.flatten_indices() if hasattr(ds, "flatten_indices") else ds
    except Exception:
        return ds


def upload_to_hub(dataset: Dataset, repo_id: str, split: str = "train"):
    logger.info(f"Uploading to HF Hub repo: {repo_id} ...")
    dataset.push_to_hub(repo_id, split=split)
    logger.info(f"Successfully pushed to {repo_id}")


def save_or_push_from_jsonl(jsonl_path, save_loc=None, hf_repo=None):
    ds = load_dataset("json", data_files=jsonl_path, split="train")
    if hf_repo:
        logger.info(f"Uploading to HF Hub repo: {hf_repo} ...")
        ds.push_to_hub(hf_repo, split="train")
        logger.info(f"Successfully pushed to {hf_repo}")
    elif save_loc:
        os.makedirs(save_loc, exist_ok=True)
        ds.save_to_disk(save_loc)
        logger.info(f"Saved to {save_loc}")


def load_manifest(path=MANIFEST_FILE):
    if os.path.exists(path):
        return set(json.load(open(path)))
    return set()


def save_manifest(completed, path=MANIFEST_FILE):
    with open(path, "w") as f:
        json.dump(list(completed), f)


def safe_append_jsonl(ds, output_path):
    with open(output_path, "a") as f:
        for ex in ds:
            f.write(json.dumps(ex) + "\n")


def upload_parquet_to_hub(ds: Dataset, shard_name: str, repo_id: str, max_retries=10):
    for attempt in range(max_retries):
        try:
            logger.info(f"Uploading {shard_name} to {repo_id}")
            ds.push_to_hub(repo_id, split=shard_name)
            logger.info(f"Uploaded {shard_name}")
            return
        except Exception as e:
            logger.warning(f"Upload attempt {attempt + 1} failed for {shard_name}: {e}")
            if attempt < max_retries - 1:
                sleep_time = random.uniform(90, 180) * (attempt + 1)
                logger.info(f"Retrying after {sleep_time:.1f}s...")
                time.sleep(sleep_time)
            else:
                logger.error(f"Final failure for {shard_name} after {max_retries} retries")


# Parquet save + upload
def save_and_upload_parquet(ds, dataset_name, completed_set, lang=None):
    name = f"{dataset_name}_{lang.split('-')[0]}" if lang else dataset_name
    if name in completed_set:
        logger.info(f"Skipping already done: {name}")
        return name

    try:
        shard_path = os.path.join(OUTPUT_DIR, f"{name}.parquet")
        ds.to_parquet(shard_path)
        logger.info(f"Saved shard: {shard_path}")
        upload_parquet_to_hub(ds, shard_name=name, repo_id=HF_REPO)
        return name
    except Exception as e:
        logger.warning(f"Failed processing {name}: {e}")
        return None
