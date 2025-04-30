from datasets import load_dataset, Features, Value, Dataset
import json
import pandas as pd
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import argparse
from random import random, seed
import os
from tqdm import tqdm
from huggingface_hub import list_repo_files

from utils import save_manifest, safe_append_jsonl, safe_load, safe_map, save_or_push_from_jsonl
from utils import flatten_if_needed, load_manifest, save_and_upload_parquet

import fsspec
import pyarrow.parquet as pq

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
seed(42)

glot_langs = ['knv_Latn', 'tgk_Latn', 'ton_Latn', 'nld_Latn', 'tzo_Latn', 'cuk_Latn', 'fil_Latn', 'hau_Arab', 'uzb_Cyrl', 'jav_Latn', 'rap_Latn', 'bak_Cyrl', 'por_Latn', 'hbo_Hebr', 'quy_Latn', 'hnj_Latn', 'ast_Latn', 'cos_Latn', 'fon_Latn', 'sna_Latn', 'dzo_Tibt', 'nob_Latn', 'nch_Latn', 'che_Cyrl', 'ext_Latn', 'dtp_Latn', 'yue_Hani', 'kbd_Cyrl', 'mar_Deva', 'ron_Latn', 'acr_Latn', 'afb_Arab', 'sqi_Latn', 'eng_Latn', 'ksd_Latn', 'bcl_Latn', 'ksh_Latn', 'hin_Latn', 'myv_Cyrl', 'kjh_Cyrl', 'sah_Cyrl', 'naq_Latn', 'tdt_Latn', 'kac_Latn', 'cak_Latn', 'kir_Cyrl', 'mps_Latn', 'yid_Hebr', 'srn_Latn', 'div_Thaa', 'mkd_Cyrl', 'bre_Latn', 'tvl_Latn', 'ven_Latn', 'wuu_Hani', 'mwl_Latn', 'miq_Latn', 'slv_Latn', 'hrv_Latn', 'hmo_Latn', 'som_Latn', 'bod_Tibt', 'pls_Latn', 'ile_Latn', 'luo_Latn', 'pus_Arab', 'fao_Latn', 'ces_Latn', 'fas_Arab', 'swa_Latn', 'ary_Arab', 'tbz_Latn', 'hus_Latn', 'ote_Latn', 'ilo_Latn', 'abk_Cyrl', 'bqc_Latn', 'hil_Latn', 'pon_Latn', 'zul_Latn', 'als_Latn', 'pes_Arab', 'bpy_Beng', 'bos_Latn', 'sot_Latn', 'lin_Latn', 'tuk_Cyrl', 'gla_Latn', 'wln_Latn', 'apc_Arab', 'hin_Deva', 'hye_Armn', 'tir_Ethi', 'pap_Latn', 'gcf_Latn', 'cjk_Latn', 'pcd_Latn', 'tur_Latn', 'kon_Latn', 'csy_Latn', 'bul_Cyrl', 'xho_Latn', 'guc_Latn', 'aka_Latn', 'kea_Latn', 'bar_Latn', 'sme_Latn', 'csb_Latn', 'bak_Latn', 'djk_Latn', 'xav_Latn', 'oci_Latn', 'acm_Arab', 'rmy_Cyrl', 'krc_Cyrl', 'cym_Latn', 'lus_Latn', 'ngu_Latn', 'yom_Latn', 'tam_Taml', 'ajp_Arab', 'epo_Latn', 'fra_Latn', 'ita_Latn', 'seh_Latn', 'hbs_Latn', 'uzn_Cyrl', 'ksw_Mymr', 'pms_Latn', 'zlm_Latn', 'qub_Latn', 'arg_Latn', 'enm_Latn', 'kaa_Cyrl', 'toj_Latn', 'spa_Latn', 'pol_Latn', 'kos_Latn', 'kab_Latn', 'pan_Guru', 'nan_Latn', 'aze_Latn', 'ara_Arab', 'meu_Latn', 'som_Arab', 'lvs_Latn', 'nbl_Latn', 'crh_Latn', 'kbp_Latn', 'tgl_Latn', 'kmb_Latn', 'hun_Latn', 'yao_Latn', 'arn_Latn', 'jbo_Latn', 'mzn_Arab', 'lzh_Hani', 'heb_Hebr', 'bjn_Latn', 'gug_Latn', 'swc_Latn', 'yor_Latn', 'ban_Latn', 'tlh_Latn', 'chv_Cyrl', 'sin_Sinh', 'ind_Latn', 'amh_Ethi', 'zea_Latn', 'kpg_Latn', 'glk_Arab', 'crh_Cyrl', 'nyu_Latn', 'ibo_Latn', 'msa_Latn', 'prs_Arab', 'nap_Latn', 'bik_Latn', 'srp_Cyrl', 'lao_Laoo', 'kom_Cyrl', 'nde_Latn', 'hui_Latn', 'uig_Latn', 'new_Deva', 'kur_Arab', 'sco_Latn', 'ayr_Latn', 'suz_Deva', 'wal_Latn', 'mlt_Latn', 'asm_Beng', 'san_Deva', 'kaz_Cyrl', 'iba_Latn', 'tuk_Latn', 'nso_Latn', 'run_Latn', 'ctu_Latn', 'bam_Latn', 'fin_Latn', 'gor_Latn', 'kmr_Latn', 'pag_Latn', 'niu_Latn', 'xmf_Geor', 'ekk_Latn', 'lmo_Latn', 'ceb_Latn', 'mhr_Cyrl', 'plt_Latn', 'qvi_Latn', 'roh_Latn', 'aln_Latn', 'mah_Latn', 'npi_Deva', 'tok_Latn', 'mgh_Latn', 'eml_Latn', 'pnb_Arab', 'nav_Latn', 'cat_Latn', 'gym_Latn', 'sat_Olck', 'snd_Arab', 'isl_Latn', 'kal_Latn', 'aoj_Latn', 'zai_Latn', 'guj_Gujr', 'min_Latn', 'grc_Grek', 'hmn_Latn', 'ido_Latn', 'khm_Khmr', 'quh_Latn', 'ikk_Latn', 'iku_Cans', 'tat_Latn', 'bel_Cyrl', 'dyu_Latn', 'que_Latn', 'quw_Latn', 'wol_Latn', 'hne_Deva', 'zho_Hani', 'tum_Latn', 'swh_Latn', 'kua_Latn', 'ncj_Latn', 'ewe_Latn', 'hat_Latn', 'ina_Latn', 'deu_Latn', 'ahk_Latn', 'srm_Latn', 'lug_Latn', 'ach_Latn', 'rmy_Latn', 'smo_Latn', 'mos_Latn', 'srd_Latn', 'ltz_Latn', 'srp_Latn', 'azb_Arab', 'aze_Arab', 'ori_Orya', 'mzh_Latn', 'kur_Latn', 'wbm_Latn', 'crs_Latn', 'ada_Latn', 'hif_Latn', 'jpn_Japn', 'pcm_Latn', 'tso_Latn', 'nor_Latn', 'bsb_Latn', 'gaa_Latn', 'ukr_Cyrl', 'mon_Latn', 'nep_Deva', 'guj_Deva', 'pis_Latn', 'lhu_Latn', 'nya_Latn', 'poh_Latn', 'nnb_Latn', 'grn_Latn', 'mco_Latn', 'ory_Orya', 'ful_Latn', 'diq_Latn', 'sag_Latn', 'afr_Latn', 'haw_Latn', 'umb_Latn', 'hsb_Latn', 'fij_Latn', 'hbs_Cyrl', 'san_Latn', 'vls_Latn', 'zsm_Latn', 'lij_Latn', 'quc_Latn', 'mam_Latn', 'tls_Latn', 'tuc_Latn', 'dan_Latn', 'rue_Cyrl', 'ace_Latn', 'bem_Latn', 'kam_Latn', 'kaa_Latn', 'ndo_Latn', 'oss_Cyrl', 'lit_Latn', 'frr_Latn', 'yap_Latn', 'bzj_Latn', 'gom_Latn', 'swe_Latn', 'lfn_Latn', 'cmn_Hani', 'mon_Cyrl', 'vep_Latn', 'ixl_Latn', 'gil_Latn', 'mau_Latn', 'tsn_Latn', 'aym_Latn', 'vec_Latn', 'gom_Deva', 'fur_Latn', 'kin_Latn', 'gcr_Latn', 'sgs_Latn', 'bih_Deva', 'vie_Latn', 'tha_Thai', 'pau_Latn', 'est_Latn', 'lue_Latn', 'rug_Latn', 'kjb_Latn', 'kik_Latn', 'mri_Latn', 'ber_Latn', 'ssw_Latn', 'cab_Latn', 'quz_Latn', 'arb_Arab', 'mai_Deva', 'bew_Cyrl', 'tat_Cyrl', 'mya_Mymr', 'alt_Cyrl', 'nno_Latn', 'hrx_Latn', 'hau_Latn', 'gsw_Latn', 'pam_Latn', 'sun_Latn', 'lat_Latn', 'bis_Latn', 'udm_Cyrl', 'tca_Latn', 'uig_Arab', 'glg_Latn', 'tah_Latn', 'ckb_Arab', 'gle_Latn', 'lim_Latn', 'slk_Latn', 'nds_Latn', 'kor_Hang', 'uzb_Latn', 'pfl_Latn', 'azj_Latn', 'tgk_Cyrl', 'glv_Latn', 'jam_Latn', 'kat_Geor', 'fry_Latn', 'kat_Latn', 'twi_Latn', 'eus_Latn', 'toi_Latn', 'mlg_Latn', 'tyv_Cyrl', 'arz_Arab', 'hyw_Armn', 'chk_Latn', 'vol_Latn', 'kek_Latn', 'teo_Latn', 'ell_Grek', 'kan_Knda', 'tpi_Latn', 'rop_Latn', 'lua_Latn', 'mad_Latn', 'top_Latn', 'scn_Latn', 'war_Latn', 'ngl_Latn', 'mal_Mlym', 'szl_Latn', 'orm_Latn', 'urd_Arab', 'cbk_Latn', 'tgk_Arab']
cv_langs = ['ab', 'af', 'am', 'ar', 'as', 'ast', 'az', 'ba', 'bas', 'be', 'bg', 'bn', 'br', 'ca', 'ckb', 'cnh', 'cs', 'cv', 'cy', 'da', 'de', 'dv', 'dyu', 'el', 'en', 'eo', 'es', 'et', 'eu', 'fa', 'fi', 'fr', 'fy-NL', 'ga-IE', 'gl', 'gn', 'ha', 'he', 'hi', 'hsb', 'ht', 'hu', 'hy-AM', 'ia', 'id', 'ig', 'is', 'it', 'ja', 'ka', 'kab', 'kk', 'kmr', 'ko', 'ky', 'lg', 'lij', 'lo', 'lt', 'ltg', 'lv', 'mdf', 'mhr', 'mk', 'ml', 'mn', 'mr', 'mrj', 'mt', 'myv', 'nan-tw', 'ne-NP', 'nhi', 'nl', 'nn-NO', 'nso', 'oc', 'or', 'os', 'pa-IN', 'pl', 'ps', 'pt', 'quy', 'rm-sursilv', 'rm-vallader', 'ro', 'ru', 'rw', 'sah', 'sat', 'sc', 'sk', 'skr', 'sl', 'sq', 'sr', 'sv-SE', 'sw', 'ta', 'te', 'th', 'ti', 'tig', 'tk', 'tok', 'tr', 'tt', 'tw', 'ug', 'uk', 'ur', 'uz', 'vi', 'vot', 'yi', 'yo', 'yue', 'zgh', 'zh-CN', 'zh-HK', 'zh-TW', 'zu', 'zza']
qed_amara = [
    "ab", "ae", "aeb", "af", "ak", "am", "an", "ar", "arq", "arz", "as", "ase", "ast", "av", "ay", "az",
    "ba", "bm", "be", "ber", "bg", "bh", "bi", "bn", "bnt", "bo", "br", "bs", "bug", "ca", "ce", "ceb",
    "ch", "cho", "cku", "cnh", "co", "cr", "cs", "cu", "cv", "cy", "da", "de", "dv", "dz", "ee", "efi",
    "el", "en", "eo", "es", "et", "eu", "fa", "ff", "fi", "fil", "fj", "fo", "fr", "ff", "ga", "gd", "gl",
    "gn", "gu", "hai", "ha", "haw", "haz", "hb", "hch", "he", "hi", "ho", "hr", "ht", "hu", "hup", "hus",
    "hy", "hz", "ia", "ig", "id", "ie", "ik", "inh", "io", "iro", "is", "it", "iu", "ja", "jv", "ka",
    "kar", "kr", "ki", "rw", "kj", "kk", "kl", "km", "kn", "ko", "ksh", "ku", "kv", "kw", "ky", "la",
    "lb", "lg", "li", "ln", "lkt", "lld", "lo", "lt", "ltg", "lu", "luo", "luy", "lv", "mad", "mfe",
    "mi", "mk", "ml", "mg", "mn", "mni", "ro", "moh", "mos", "mr", "ms", "mt", "mus", "my", "nb", "nci",
    "nd", "ne", "nl", "nn", "nso", "nv", "ny", "oc", "or", "om", "pam", "pa", "pap", "pi", "pl", "pnb",
    "prs", "ps", "pt", "qu", "rm", "ro", "ru", "rn", "rup", "ry", "sa", "sc", "scn", "sco", "sd", "sg",
    "sgn", "sh", "si", "sk", "sl", "sm", "sn", "so", "st", "sq", "sr", "sr", "sv", "sw", "szl", "ta",
    "te", "tet", "tg", "th", "ti", "tk", "tl", "tlh", "to", "tr", "ts", "tt", "tw", "ug", "uk", "umb",
    "ur", "uz", "ve", "vi", "vls", "vo", "wa", "wo", "xh", "yaq", "yi", "yo", "za", "zam", "zh", "zu"
]

ccaligned_langs = [
    "af_ZA", "ak_GH", "am_ET", "ar_AR", "as_IN", "ay_BO", "az_AZ", "az_IR", "be_BY", "bg_BG",
    "bm_ML", "bn_IN", "br_FR", "bs_BA", "ca_ES", "cb_IQ", "cs_CZ", "cx_PH", "cy_GB", "da_DK",
    "de_DE", "el_GR", "es_XX", "et_EE", "fa_IR", "ff_NG", "fi_FI", "fr_XX", "gu_IN", "ha_NG",
    "he_IL", "hi_IN", "hr_HR", "ht_HT", "hu_HU", "hy_AM", "id_ID", "ig_NG", "is_IS", "it_IT",
    "ja_XX", "jv_ID", "ka_GE", "kg_AO", "kk_KZ", "km_KH", "kn_IN", "ko_KR", "ku_TR", "ky_KG",
    "lg_UG", "ln_CD", "lo_LA", "lt_LT", "lv_LV", "mg_MG", "mi_NZ", "mk_MK", "ml_IN", "mn_MN",
    "mr_IN", "ms_MY", "mt_MT", "my_MM", "ne_NP", "nl_XX", "no_XX", "ns_ZA", "ny_MW", "om_KE",
    "or_IN", "pa_IN", "pl_PL", "ps_AF", "pt_XX", "qa_MM", "qd_MM", "ro_RO", "ru_RU", "si_LK",
    "sk_SK", "sl_SI", "sn_ZW", "so_SO", "sq_AL", "sr_RS", "ss_SZ", "st_ZA", "su_ID", "sv_SE",
    "sw_KE", "sz_PL", "ta_IN", "te_IN", "tg_TJ", "th_TH", "ti_ET", "tl_XX", "tn_BW", "tr_TR",
    "ts_ZA", "tz_MA", "uk_UA", "ur_PK", "ve_ZA", "vi_VN", "wo_SN", "xh_ZA", "yo_NG", "zh_CN",
    "zh_TW", "zu_ZA", "zz_TR"
]
minds_langs = ['cs-CZ', 'de-DE', 'en-AU', 'en-GB', 'en-US', 'es-ES', 'fr-FR', 'it-IT', 'ko-KR', 'nl-NL', 'pl-PL',
               'pt-PT', 'ru-RU', 'zh-CN']

afrisenti_langs = ['amh', 'arq', 'ary', 'eng', 'hau', 'ibo', 'kin', 'orm', 'pcm', 'por', 'swa', 'tir', 'tso', 'twi', 'yor']
naijasenti_langs = ['hau', 'ibo', 'pcm', 'yor']
voxpop_df = pd.read_csv('langs/voxpopuli_langs.csv')
voxpop_langs = ['en', 'de', 'fr', 'es', 'pl', 'it', 'ro', 'hu', 'cs', 'nl', 'fi', 'hr', 'sk', 'sl', 'et', 'lt']


bloom_lang_df = pd.read_csv('langs/bloom_languages.csv')
bloom_langs = dict(zip(bloom_lang_df['ISO 639-3'], bloom_lang_df['Name']))
bloom_lang_keys = bloom_langs.keys()


iso_codes = pd.read_csv('langs/iso_code_mapping.csv')
iso_1_to_2 = dict(zip(iso_codes['iso_1'], iso_codes['iso_3']))

OUTPUT_FILE = "final_lid_data.jsonl"
MANIFEST_FILE = "completed.json"


def format_tatoeba_pair_thread(pair, dataset_name="tatoeba"):
    lang1, lang2 = pair
    try:
        ds = load_dataset("Helsinki-NLP/tatoeba", lang1=lang1, lang2=lang2, split="train", trust_remote_code=True)
        formatted = []
        for ex in ds:
            tr = ex["translation"]
            if "en" in [lang1, lang2]:
                non_en = lang2 if lang1 == "en" else lang1
                formatted.append({
                    "sentence": tr[non_en],
                    "data_lang": non_en,
                    "language": iso_1_to_2.get(non_en, non_en),
                    "dataset": dataset_name
                })
            else:
                formatted.extend([
                    {"sentence": tr[lang1], "data_lang": lang1, "language": iso_1_to_2.get(lang1, lang1), "dataset": dataset_name},
                    {"sentence": tr[lang2], "data_lang": lang1, "language": iso_1_to_2.get(lang2, lang2), "dataset": dataset_name}
                ])
        return formatted
    except Exception as e:
        return []


def load_validated_pairs_threaded(path):
    with open(path) as f:
        pairs = json.load(f)

    seen_langs = set()
    final_pairs = []
    for pair in pairs:
        l1, l2 = pair["lang1"], pair["lang2"]
        if "en" in (l1, l2):
            non_en = l2 if l1 == "en" else l1
            if non_en not in seen_langs:
                seen_langs.add(non_en)
                final_pairs.append((l1, l2))
        else:
            if l1 not in seen_langs and l2 not in seen_langs:
                seen_langs.update([l1, l2])
                final_pairs.append((l1, l2))
    return final_pairs


def threadpool_load_tatoeba_data(pair_file):
    pairs = load_validated_pairs_threaded(pair_file)
    results = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(format_tatoeba_pair_thread, pair): pair for pair in pairs}
        for future in as_completed(futures):
            res = future.result()
            if isinstance(res, list):
                results.extend(res)

    return results


def try_load_en_pair(lang, dataset_name):
    try:
        ds = load_dataset(dataset_name, lang1="en", lang2=lang, split='train', trust_remote_code=True, streaming=True)
        dataset = dataset_name.split('/')[-1]
        if dataset == 'tatoeba':
            return [
                {"sentence": ex[lang], "data_lang": lang, "language": iso_1_to_2.get(lang, lang), "dataset": dataset}
                for ex in ds
            ]
        else:
            return [
                {"sentence": ex["translation"][lang], "data_lang": lang, "language": iso_1_to_2.get(lang, lang),
                 "dataset": dataset}
                for ex in ds
            ]
    except (FileNotFoundError, KeyError, ValueError) as e:
        logger.warning(f"⚠️ Could not load en-{lang}: {e}")
        return []
    except Exception as e:
        logger.warning(f"⚠️ Unexpected failure en-{lang}: {e}")
        return []


def try_load_non_en_pair(lang1, lang2, dataset_name):
    try:
        ds = load_dataset(dataset_name, lang1=lang1, lang2=lang2, split='train', trust_remote_code=True, streaming=True)
        dataset = dataset_name.split('/')[-1]
        if dataset == "tatoeba":
            print(dataset)
            return [
                {"sentence": ex[lang1], "data_lang": lang1, "language": iso_1_to_2.get(lang1, lang1), "dataset": dataset}
                for ex in ds
            ] + [
                {"sentence": ex[lang2], "data_lang": lang2, "language": iso_1_to_2.get(lang2, lang2), "dataset": dataset}
                for ex in ds
            ]
        else:
            return [
                {"sentence": ex["translation"][lang1], "data_lang": lang1, "language": iso_1_to_2.get(lang1, lang1),
                 "dataset": dataset}
                for ex in ds
            ] + [
                {"sentence": ex["translation"][lang2], "data_lang": lang2, "language": iso_1_to_2.get(lang2, lang2),
                 "dataset": dataset}
                for ex in ds
            ]
    except (FileNotFoundError, KeyError, ValueError) as e:
        logger.warning(f"⚠️ Could not load {lang1}-{lang2}: {e}")
        return []
    except Exception as e:
        logger.warning(f"⚠️ Unexpected failure {lang1}-{lang2}: {e}")
        return []


def load_bilingual_data(dataset_name, lang_list):
    covered = set()
    result = []
    valid_iso1 = set(iso_codes['iso_1'].dropna())

    lang_list = [lang for lang in lang_list if len(lang) in (2, 3) and (lang in valid_iso1 or len(lang) == 3)]

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(try_load_en_pair, lang, dataset_name): lang
            for lang in lang_list if lang != "en"
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Loading EN pairs for {dataset_name}"):
            lang = futures[future]
            data = future.result()
            if data:
                result.extend(data)
                covered.add(lang)

        # Non-English pairs
        remaining = [l for l in lang_list if l not in covered and l != "en"]
        non_en_pairs = [(l1, l2) for l1, l2 in product(remaining, repeat=2) if l1 < l2]

        futures = {
            executor.submit(try_load_non_en_pair, l1, l2, dataset_name): (l1, l2)
            for (l1, l2) in non_en_pairs
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Loading non-EN pairs for {dataset_name}"):
            l1, l2 = futures[future]
            data = future.result()
            if data:
                result.extend(data)
                covered.update([l1, l2])

    return result


def load_minds14_data(lang, dataset_name="minds14"):
    ds = load_dataset("PolyAI/minds14", lang, split="train", trust_remote_code=True, download_mode="reuse_dataset_if_exists")
    return ds.map(lambda x: {
        "sentence": x["transcription"],
        "data_lang": lang,
        "language": iso_1_to_2[lang.split('-')[0]], #convert to 3 code
        "dataset": dataset_name
    }, remove_columns=ds.column_names)


def load_ted_multi_data(example, dataset_name="ted_multi"):
    sentences = []
    for lang, trans in zip(example["language"], example["translation"]):
        sentences.append({
            "sentence": trans,
            "data_lang": lang,
            "language": iso_1_to_2[lang.split('-')[0]],
            "dataset": dataset_name
        })
    return sentences


def load_ccaligned_data(lang, dataset_name="ccaligned"):
    ds = load_dataset("ahelk/ccaligned_multilingual", language_code=lang, type="sentences", split='train',
                      trust_remote_code=True, download_mode="reuse_dataset_if_exists")
    return ds.map(lambda x: {
        "sentence": x["translation"][lang],
        "data_lang": lang,
        "language": iso_1_to_2[lang.split('_')[0]],
        "dataset": dataset_name
    }, remove_columns=ds.column_names)


def load_glot500_data(lang, dataset_name="glot500"):
    ds = load_dataset("cis-lmu/Glot500", lang, split='train', trust_remote_code=True, download_mode="reuse_dataset_if_exists")
    return ds.map(lambda x: {
        "sentence": x["text"],
        "data_lang": lang,
        "language": lang.split('_')[0],
        "dataset": dataset_name
    }, remove_columns=ds.column_names)


def load_bloom_data(lang, dataset_name="bloom_stories", split='train'):
    ds = load_dataset("sil-ai/bloom-lm", lang, split=split, trust_remote_code=True, download_mode="reuse_dataset_if_exists")
    return ds.map(lambda x: {
        "sentence": x["text"],
        "data_lang": lang,
        "language": lang,
        "dataset": dataset_name
    }, remove_columns=ds.column_names)


def load_cv_data(lang, dataset_name="common_voice", split="train"):
    ds = load_dataset("mozilla-foundation/common_voice_12_0", lang, split=split, trust_remote_code=True, download_mode="reuse_dataset_if_exists")
    return ds.map(lambda x: {
        "sentence": x["sentence"],
        "data_lang": lang,
        "language": lang,
        "dataset": dataset_name
    }, remove_columns=ds.column_names)


def load_afrisenti_data(lang, dataset_name="afrisenti", split='train'):
    ds = load_dataset("masakhane/afrisenti", lang, split=split, trust_remote_code=True, download_mode="reuse_dataset_if_exists")
    return ds.map(lambda x: {
        "sentence": x["tweet"],
        "data_lang": lang,
        "language": lang,
        "dataset": dataset_name
    }, remove_columns=ds.column_names)


def load_naijasenti_data(lang, dataset_name="naijasenti", split='train'):
    ds = load_dataset("HausaNLP/NaijaSenti-Twitter", lang, split=split, trust_remote_code=True, download_mode="reuse_dataset_if_exists")
    return ds.map(lambda x: {
        "sentence": x["tweet"],
        "data_lang": lang,
        "language": lang,
        "dataset": dataset_name
    }, remove_columns=ds.column_names)


def load_cat_youtube_data(dataset_name="catalan-youtube", lang='cat', split='train'):
    ds = load_dataset("softcatala/catalan-youtube-speech", split=split, trust_remote_code=True, download_mode="reuse_dataset_if_exists")
    lang = 'cat'
    return ds.map(lambda x: {
        "sentence": x["candidate_1"],
        "data_lang": "catalan",
        "language": lang,
        "dataset": dataset_name
    }, remove_columns=ds.column_names)


def load_yue_youtube_data(dataset_name="cantonese-youtube", lang='yue', split='train'):
    ds = load_dataset("alvanlii/cantonese-youtube", split=split, streaming=True, trust_remote_code=True)

    buffer = []
    all_examples = []
    batch_size = 5000

    for ex in tqdm(ds, desc="Streaming Cantonese YouTube"):
        text = ex.get("transcript_whisper", "")
        if not text:
            continue

        buffer.append({
            "sentence": text.strip(),
            "data_lang": "cantonese",
            "language": lang,
            "dataset": dataset_name
        })

        if len(buffer) >= batch_size:
            all_examples.extend(buffer)
            buffer.clear()

    if buffer:
        all_examples.extend(buffer)

    if not all_examples:
        return None

    return Dataset.from_list(all_examples)


def load_voxpupuli_data(lang, dataset_name="voxpupuli", split='train'):
    features = Features({
        "normalized_text": Value("string"),
        "speaker_id": Value("string"),
        "language": Value("string"),
        "path": Value("string"),
        "audio": {"path": Value("string"), "array": Value("binary"), "sampling_rate": Value("int32")}
    })

    ds = load_dataset(
        "facebook/voxpopuli", lang,
        split=split,
        trust_remote_code=True,
        features=features,  # <-- FORCE custom features!
        download_mode="reuse_dataset_if_exists"
    )
    mapped_lang = iso_1_to_2[lang]
    print(lang, mapped_lang)
    return ds.map(lambda x: {
        "sentence": x["normalized_text"],
        "data_lang": lang,
        "language": mapped_lang,
        "dataset": dataset_name
    }, remove_columns=ds.column_names)


def should_keep_openlid_example(source):
    filter_sources = {"tatoeba", "pali"}
    downsample_sources = {"wili2018", "setimes", "xlsum", "leipzig", "mizan", "tep", "voa"}
    if source in filter_sources:
        return False
    if source in downsample_sources:
        return random() < 0.3  # Keep 30% of these
    return True


def load_openlid_data():
    from collections import defaultdict
    completed = load_manifest()
    dataset_name = "openlid"

    if dataset_name in completed:
        logger.info(f"✓ Skipping OpenLID: already completed.")
        return

    logger.info("Streaming OpenLID dataset...")
    ds = load_dataset("laurievb/OpenLID-v2", split="train", streaming=True, trust_remote_code=True)

    grouped = defaultdict(list)
    count = 0

    for ex in tqdm(ds, desc="Streaming OpenLID examples"):
        if not ex.get("text") or not ex.get("language"):
            continue
        if not should_keep_openlid_example(ex.get("source", "")):
            continue

        grouped[ex['language']].append({
            "sentence": ex["text"].strip(),
            "data_lang": ex["language"],
            "language": ex["language"].split('_')[0],
            "dataset": f"openlid_{ex['source']}"
        })
        count += 1

        # Save + Upload when enough examples are accumulated
        if count % 500_000 == 0:
            logger.info(f"Saving intermediate OpenLID shards at {count} examples...")
            save_grouped_shards(grouped, dataset_name, completed)
            grouped.clear()  # reset

    # Save any leftover after streaming is done
    if grouped:
        save_grouped_shards(grouped, dataset_name, completed)

    completed.add(dataset_name)
    save_manifest(completed)
    logger.info("✅ Finished processing OpenLID!")


def save_grouped_shards(grouped, dataset_name, completed_set):
    for lang, examples in tqdm(grouped.items(), desc="Uploading OpenLID shards", leave=False):
        shard_ds = Dataset.from_list(examples)
        save_and_upload_parquet(shard_ds, dataset_name, completed_set, lang)


def process_dataset_lang_pair(loader_fn, dataset_name, completed_set, lang=None):
    try:
        ds = loader_fn(lang) if lang else loader_fn()
        if ds:
            logger.info(f"Saving a uploading {dataset_name}")
            return save_and_upload_parquet(ds, dataset_name, completed_set, lang)
    except Exception as e:
        logger.warning(f"⚠️ Error loading {dataset_name} {lang}: {e}")
    return None


def run_custom(loader_fn, dataset_name, completed_set, *args, **kwargs):
    name = dataset_name
    if name in completed_set:
        logger.info(f"✓ Skipping already processed: {name}")
        return

    try:
        logger.info(f"⏳ Processing custom loader: {name}")
        data = loader_fn(*args, **kwargs)
        if not data:
            logger.warning(f"⚠️ No data returned for {name}")
            return

        # Group by language for splitting
        from collections import defaultdict
        grouped = defaultdict(list)
        for row in data:
            grouped[row['data_lang']].append(row)

        for lang, examples in tqdm(grouped.items(), desc=f"{dataset_name} (custom)", total=len(grouped)):
            ds = Dataset.from_list(examples)
            save_and_upload_parquet(ds, dataset_name, completed_set, lang)

        completed_set.add(name)
        save_manifest(completed_set)

    except Exception as e:
        logger.warning(f"Custom loader failed for {name}: {e}")


def run_parallel_jobs(loader_fn, dataset_name, lang_list=None):
    completed = load_manifest()
    futures = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        if lang_list is not None:
            all_langs = [l for l in lang_list if f"{dataset_name}_{l}" not in completed]
            for lang in all_langs:
                futures.append(executor.submit(process_dataset_lang_pair, loader_fn, dataset_name, completed, lang))
        else:
            if f"{dataset_name}" not in completed:
                futures.append(executor.submit(process_dataset_lang_pair, loader_fn, dataset_name, completed))

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"{dataset_name}"):
            name = future.result()
            if name:
                completed.add(name)

    save_manifest(completed)


def run_dataset_grouped_by_lang(ds, dataset_name, completed_set):
    if not ds:
        logger.warning(f"No dataset to process for {dataset_name}")
        return

    from collections import defaultdict
    grouped = defaultdict(list)
    for row in ds:
        grouped[row['data_lang']].append(row)

    for lang, examples in tqdm(grouped.items(), desc=f"{dataset_name} (custom)", total=len(grouped)):
        name = f"{dataset_name}_{lang}"
        if name in completed_set:
            logger.info(f"Skipping already processed: {name}")
            continue
        shard_ds = Dataset.from_list(examples)
        save_and_upload_parquet(shard_ds, dataset_name, completed_set, lang)
        completed_set.add(name)

    save_manifest(completed_set)


def load_tweetlid_data(dataset_name="tweetlid"):
    name = dataset_name
    completed = load_manifest()
    if name in completed:
        logger.info(f"Skipping already processed: {name}")
        return

    try:
        logger.info(f"Loading and uploading {name}")
        df = pd.read_csv('../dataset/full_tweetlid_train.txt', sep='\t', header=None, names=['id', 'user', 'language', 'text'])
        df = df[~df['language'].str.contains(r'[\+\/]', regex=True)]

        records = [
            {
                "sentence": str(row["text"]).strip(),
                "data_lang": row["language"],
                "language": iso_1_to_2[row["language"]],
                "dataset": dataset_name
            }
            for _, row in df.iterrows()
        ]

        if not records:
            logger.warning(f"⚠️ No data to upload for {name}")
            return

        ds = Dataset.from_list(records)

        save_and_upload_parquet(ds, dataset_name, completed)

        completed.add(name)
        save_manifest(completed)

    except Exception as e:
        logger.warning(f"Failed processing {name}: {e}")


def main(data_group):
    completed = load_manifest()
    if data_group == "minds14":
        run_parallel_jobs(load_minds14_data, "minds14", minds_langs)

    elif data_group == "glot500":
        run_parallel_jobs(load_glot500_data, "glot500", glot_langs)

    elif data_group == "bloom":
        run_parallel_jobs(load_bloom_data, "bloom_stories", bloom_lang_keys)

    elif data_group == "common_voice":
        run_parallel_jobs(loader_fn=load_cv_data, dataset_name="common_voice", lang_list=cv_langs),

    elif data_group == "afrisenti":
        run_parallel_jobs(load_afrisenti_data, "afrisenti", afrisenti_langs)

    elif data_group == "naijasenti":
        run_parallel_jobs(load_naijasenti_data, "naijasenti", naijasenti_langs)

    elif data_group == "voxpopuli":
        run_parallel_jobs(load_voxpupuli_data, "voxpopuli", voxpop_langs)

    elif data_group == "ccaligned":
        run_parallel_jobs(load_ccaligned_data, "ccaligned", ccaligned_langs)

    elif data_group == "openlid":
        run_parallel_jobs(load_openlid_data, "openlid")

    elif data_group == "tweetlid":
        run_parallel_jobs(load_tweetlid_data, "tweetlid")

    elif data_group == "youtube":
        run_parallel_jobs(load_yue_youtube_data, "cantonese-youtube", ["yue"])
        run_parallel_jobs(load_cat_youtube_data, "catalan-youtube", ["cat"])

    elif data_group == "bilingual_qed":
        run_custom(load_bilingual_data, "qed_amara", completed, 'Helsinki-NLP/qed_amara', qed_amara)
        run_custom(threadpool_load_tatoeba_data, "tatoeba", completed, "valid_tatoeba_pairs.json")
    elif data_group == "bilingual_open":
        run_custom(load_bilingual_data, "open_subtitles", completed, 'Helsinki-NLP/open_subtitles', qed_amara)

    elif data_group == "ted_multi":
        ds = safe_load(load_dataset, "neulab/ted_multi", split='train', trust_remote_code=True)
        if ds:
            all_rows = []
            for ex in tqdm(ds['translations'], desc="Mapping TED Multi"):
                rows = load_ted_multi_data(ex)
                all_rows.extend(rows)
            run_dataset_grouped_by_lang(all_rows, "ted_multi", completed)

    else:
        logger.error(f"Unknown group: {data_group}")


def parse_args():
    parser = argparse.ArgumentParser(description="LID Data Curation")
    parser.add_argument("--group", type=str, required=True,
                        help="Dataset group to process, e.g., minds14, glot500, ted_multi, bilingual")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_group = args.group
    main(data_group)
