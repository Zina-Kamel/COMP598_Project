import os
import re
import string
import numpy as np
import pandas as pd
from datasets import load_dataset
import gcld3
import langid
import fasttext
from pyfranc import franc
from huggingface_hub import hf_hub_download

from langdetect import detect
from langdetect import detect_langs

from langdetect import DetectorFactory
DetectorFactory.seed = 0

model_names = "laurievb/OpenLID", "cis-lmu/glotlid", "facebook/fasttext-language-identification"
smol_langs = [
    "aa", "ab", "ace", "ach", "ady", "aeb", "af", "ahr", "aii", "ak", "alz", "am", "apc", "apd", "ar", "arn", "arz",
    "as", "av", "awa", "ay", "ayl", "ba", "bal", "ban", "bbc", "bci", "bem", "ber", "bew", "bfq", "bfy", "bgq", "bho",
    "bik", "bjn", "bm", "bns", "bo", "br", "bra", "brx", "bts", "btx", "bua", "bug", "ccp", "ce", "cgg", "ch", "chk",
    "chm", "ckb", "cnh", "crh", "crs", "ctg", "cv", "dhd", "din", "doi", "dov", "dv", "dyu", "dz", "ee", "efi", "en",
    "es", "fa", "ff", "fj", "fo", "fon", "fr", "fur", "gaa", "gn", "gom", "grt", "ha", "hi", "hil", "hne", "hoc", "hrx",
    "iba", "ig", "ilo", "iso", "iu", "jam", "kaa", "kac", "kbd", "kek", "kfy", "kg", "kha", "ki", "kl", "kr", "kri",
    "kru", "ks", "ktu", "kv", "lep", "lg", "li", "lif", "lij", "lmo", "ln", "ltg", "lu", "lua", "luo", "lus", "mad",
    "mag", "mai", "mak", "mam", "meo", "mfe", "mg", "mh", "min", "mjl", "mni", "mos", "ms", "mtr", "mwr", "nd", "ndc",
    "ne", "new", "nhe", "noe", "nr", "nso", "nus", "nv", "ny", "oc", "om", "os", "pa", "pag", "pam", "pap", "pcm", "pt",
    "qu", "quc", "rhg", "rn", "rom", "rw", "sa", "sah", "sat", "scl", "scn", "sd", "se", "sg", "sgj", "shn", "sjp",
    "skr", "sn", "so", "spv", "ss", "st", "sus", "sw", "syl", "szl", "tcy", "tet", "ti", "tiv", "tn", "to", "tpi", "trp",
    "ts", "tum", "ty", "tyv", "udm", "unr", "ve", "vec", "war", "wbr", "wo", "xh", "xnr", "xsr", "yo", "yua", "yue",
    "zap", "zh", "zu", "zza"
]

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


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_model(model_name):
    model_path = hf_hub_download(repo_id=model_name, filename="model.bin")
    model = fasttext.load_model(model_path)
    return model


def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'<URL>', '', text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def get_prediction(model_name, df, codeswitch=True):
    pred_lang = []
    pred_prob = []
    if model_name == "langid":
        predictions = [langid.classify(text) for text in df["codeswitch_sentence"]]
        pred_prob = [np.asarray(pred[1]) for pred in predictions]
        pred_lang = [pred[0] for pred in predictions]
    elif model_name == "langdetect":
        top_preds = []
        for text in df["codeswitch_sentence"]:
            try:
                preds = detect_langs(text.replace('\n', ''))
                top_preds.append(preds)
                pred_prob.append(str(preds[0]).split(':')[1])
                pred_lang.append(str(preds[0]).split(':')[0])
            except Exception:
                top_preds.append(None)
                pred_prob.append(None)
                pred_lang.append(None)
        df['top_pred'] = top_preds
    elif model_name == "franc":
        predictions = [franc.lang_detect(text)[:3] for text in df["codeswitch_sentence"]]
        top_preds = [{label: float(prob) for label, prob in preds}
                      for preds in predictions]
        df['top_pred'] = top_preds
        pred_prob = [float(pred[0][1]) for pred in predictions]
        pred_lang = [pred[0][0] for pred in predictions]
    elif model_name == "cld3":
        detector = gcld3.NNetLanguageIdentifier(min_num_bytes=0, max_num_bytes=1000)
        if codeswitch:
            predictions = [detector.FindTopNMostFreqLangs(text=text, num_langs=2) for text in df["codeswitch_sentence"]]
            top_preds = [
                {p.language: float(p.probability) for p in prob}
                for prob in predictions
            ]
            pred_prob = [pred[0].probability for pred in predictions]
            pred_lang = [pred[0].language for pred in predictions]
            df['top_pred'] = top_preds
        else:
            prediction = [detector.FindLanguage(text=text) for text in df["codeswitch_sentence"]]
            pred_prob = [pred.probability for pred in prediction]
            pred_lang = [pred.language for pred in prediction]
    else:
        model = load_model(model_name)
        predictions = [model.predict(text.replace('\n', ''), k=3) for text in df["codeswitch_sentence"]]
        top_preds = [{label.replace("__label__", ""): float(prob) for label, prob in zip(labels, probs)}
                     for labels, probs in predictions]
        pred_prob = [np.asarray(pred[1])[0] for pred in predictions]
        pred_lang = [pred[0][0].replace("__label__", "") for pred in predictions]
        df['top_pred'] = top_preds

    df["pred_lang"] = pred_lang
    df["pred_prob"] = pred_prob

    return df


def run_flores(model_name, output_dir="./"):
    data_name = "openlanguagedata/flores_plus"
    data = load_dataset(data_name)["devtest"]
    model = load_model(model_name)
    predictions = [model.predict(text) for text in data["text"]]
    results = pd.DataFrame({
        "domain": data["domain"],
        "topic": data["topic"],
        "text": data["text"],
        "language": data["iso_639_3"],
        "lang_script": data["iso_15924"],
        "pred_lang": [pred[0][0] for pred in predictions],
        "pred_prob": [np.asarray(pred[1])[0] for pred in predictions]
    })
    create_dir(output_dir)
    results.to_csv(f"{output_dir}{model_name.split('/')[-1]}.csv", index=False)
    return results


def run_smols(model_name, output_dir="results/"):
    data_name = "google/smol"
    model = load_model(model_name)
    result_list = []

    languages = smol_sent + smol_doc
    for lang in languages:
        try:
            data = load_dataset(data_name, f"smolsent__{lang}")["train"]
            language_text = data['trg']
        except ValueError:
            data = load_dataset(data_name, f"smoldoc__{lang}")["train"]
            language_text = [" ".join(text) for text in data['trgs']]

        clean_texts = [text.replace("\n", " ").strip() for text in language_text]
        predictions = [model.predict(text) for text in clean_texts]
        results = pd.DataFrame({
            "text": language_text,
            "language": data["tl"],
            "pred_lang": [pred[0][0] for pred in predictions],
            "pred_prob": [np.asarray(pred[1])[0] for pred in predictions]
        })
        result_list.append(results)
    create_dir(output_dir)
    result_df = pd.concat(result_list, ignore_index=True) if result_list else pd.DataFrame()
    result_df.to_csv(f"{output_dir}{model_name.split('/')[-1]}_smol.csv", index=False)
    return result_df


def run_codeswitch(model_name, output_dir="results/"):
    data = pd.read_csv('dataset/code-switch/combined_cs_datset.csv')
    data = get_prediction(model_name, data)

    create_dir(output_dir)
    data.to_csv(f"{output_dir}{model_name.split('/')[-1]}_multi_cs.csv", index=False)
    return data


run_codeswitch("facebook/fasttext-language-identification")
# model_names = "laurievb/OpenLID", "cis-lmu/glotlid", "facebook/fasttext-language-identification,
# "langid", "langdetect", "franc", "cld3"
