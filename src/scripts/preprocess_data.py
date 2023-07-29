import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
import numpy as np
import pandas as pd
from nltk.tokenize import wordpunct_tokenize

from data.preprocessing import (
    add_sentence_tokens,
    apply_default_pipeline,
    split_and_save_dataframe,
    trim_outliers,
    remove_duplicates,
)
from utils.constants import MAX_WORDS

train_proportion = 0.7
val_proportion = 0.1
test_proportion = 0.2

df = pd.read_csv("data/sampled_data.csv", header=None, names=["en", "hu"])

print(
    f"Preprocessing data with basic pipeline and truncating it to sentences with maximum length of {MAX_WORDS}..."
)

df = apply_default_pipeline(df, "en", "en_processed", lang="en")
df = apply_default_pipeline(df, "hu", "hu_processed", lang="hu")

df["en_tokens"] = df["en_processed"].apply(wordpunct_tokenize)
df["hu_tokens"] = df["hu_processed"].apply(wordpunct_tokenize)
df = trim_outliers(df, "en_tokens", MAX_WORDS)
df = trim_outliers(df, "hu_tokens", MAX_WORDS)
df = remove_duplicates(df, "en_tokens")
df = remove_duplicates(df, "hu_tokens")
df = df.drop("en_tokens", axis=1)
df = df.drop("hu_tokens", axis=1)

print(
    f"Splitting into training set ({train_proportion*100:.2f}%), validation set ({val_proportion*100:.2f}%), and test set ({test_proportion*100:.2f}%)..."
)

split_and_save_dataframe(
    df, 0.7, 0.1, 0.2, "preprocessed", "data/preprocessed/", random_state=0
)
