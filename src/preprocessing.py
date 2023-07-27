import string
import re

import pandas as pd
import numpy as np
from pathlib import Path
from num2words import num2words
from sklearn.model_selection import train_test_split

from typing import List


## Normalisation methods used for every preprocessing pipeline ##
def make_lower(s: str) -> str:
    return str(s).lower()


def strip_whitespace(s: str) -> str:
    # double spaces
    s = re.sub(r" +", " ", s)
    return s.strip()


def remove_xml_tags(xml_string):
    return re.sub(r"<[^>]+>", "", xml_string)


def remove_parantheses_strings(s: str) -> str:
    return re.sub(r"\([^()]*\)", "", s)


def remove_invalid_dash(s: str) -> str:
    return re.sub(r" - ", "", s)


def remove_punctuation(s: str) -> str:
    valid_punctuations = [",", "'", "-"]
    invalid_punctuation = string.punctuation
    for p in valid_punctuations:
        invalid_punctuation = invalid_punctuation.replace(p, "")
    replacements = {p: " " for p in invalid_punctuation}
    return str(s).translate(str.maketrans(replacements))


def numbers_to_words(s: str, lang: str = "en") -> str:
    return " ".join([num2words(x, lang=lang) if x.isdigit() else x for x in s.split()])


def remove_empty_text(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    mask = df[col_name].str.len() > 0
    return_df = df.loc[mask]
    return return_df.reset_index(drop=True)


def apply_default_pipeline(data: pd.DataFrame, in_col: str, out_col: str, lang: str):
    df = data.copy()
    language_agnostic_pipeline = [
        make_lower,
        remove_xml_tags,
        remove_parantheses_strings,
        remove_punctuation,
        remove_invalid_dash,
    ]
    df[out_col] = df[in_col].astype(str)
    for f in language_agnostic_pipeline:
        df[out_col] = df[out_col].apply(f)
    df[out_col] = df[out_col].apply(numbers_to_words, lang=lang)
    df[out_col] = df[out_col].apply(strip_whitespace)
    df = remove_empty_text(df, out_col)
    return df


def get_split_idx(
    dataset_length: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    **kwargs
):
    assert train_ratio + val_ratio + test_ratio == 1
    relative_val_size = val_ratio / (1 - test_ratio)
    full_idx = np.arange(0, dataset_length)
    train_val_idx, test_idx, y_trainval, _ = train_test_split(
        full_idx, full_idx, test_size=test_ratio, **kwargs
    )
    train_idx, val_idx, _, _ = train_test_split(
        train_val_idx, y_trainval, test_size=relative_val_size, **kwargs
    )
    return train_idx, val_idx, test_idx


def split_and_save_dataframe(
    data: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    file_base_name: str,
    out_dir: str,
    **kwargs
):
    train_idx, val_idx, test_idx = get_split_idx(
        len(data), train_ratio, val_ratio, test_ratio, **kwargs
    )
    train_df = data.iloc[train_idx].reset_index(drop=True)
    val_df = data.iloc[val_idx].reset_index(drop=True)
    test_df = data.iloc[test_idx].reset_index(drop=True)
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir + file_base_name + "_train.csv")
    val_df.to_csv(out_dir + file_base_name + "_val.csv")
    test_df.to_csv(out_dir + file_base_name + "_test.csv")


def get_vocab(sentences: List[List]):
    text = []
    for sent in sentences:
        text += sent
    return set(text)

def word2idx(vocab: List[str]):
    w2i = {"<SOS>": 0, "<EOS>": 1}
    n_words = 2
    for word in vocab:
        if word == "<SOS>" or word == "<EOS>":
            continue
        w2i[word] = n_words
        n_words += 1
    return w2i

def sent2idx(word2idx, sentence, max_len):
    idxs = np.zeros(max_len, dtype=np.int32)
    word_idxs = [word2idx[word] for word in sentence]
    word_idxs.append(word2idx["<EOS>"])
    idxs[:len(word_idxs)] = word_idxs
    return idxs

def add_sentence_tokens(s: list):
    return ["<SOS>"] + s + ["<EOS>"]