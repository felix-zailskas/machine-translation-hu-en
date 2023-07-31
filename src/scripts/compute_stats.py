import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import matplotlib.pyplot as plt
import pandas as pd
from nltk.tokenize import wordpunct_tokenize
from utils.text_statistics import (
    get_sentence_lengths,
    get_list_of_unique_tokens,
    get_list_of_unique_chars,
)
from utils.constants import MAX_WORDS
from data.preprocessing import apply_default_pipeline, trim_outliers, remove_duplicates

df = pd.read_csv("data/sampled_data.csv", header=None, names=["en", "hu"])

df = apply_default_pipeline(df, "en", "en_processed", lang="en")
df = apply_default_pipeline(df, "hu", "hu_processed", lang="hu")

en_lens = get_sentence_lengths(df["en_processed"])
en_toks = get_list_of_unique_tokens(df["en_processed"])
en_chars = get_list_of_unique_chars(df["en_processed"])
hu_lens = get_sentence_lengths(df["hu_processed"])
hu_toks = get_list_of_unique_tokens(df["hu_processed"])
hu_chars = get_list_of_unique_chars(df["hu_processed"])
diff_lens = en_lens - hu_lens

plt.hist(en_lens, bins=50, alpha=0.45, color="red")
plt.hist(hu_lens, bins=50, alpha=0.45, color="blue")

plt.title(
    "Distribution of sentence lengths for each language\nbefore limiting sentence length"
)

plt.legend(["English", "Hungarian"])

plt.show()

print(
    f"""
------------------------------------------------
BEFORE LIMITING SENTENCE LENGTH
Dataset contains a total of {len(df)} entries.

English Sentence lengths in Tokens:
    Max:        {en_lens.max()}
    Min:        {en_lens.min()}
    Mean:       {en_lens.mean()}
    #Tokens:    {len(en_toks)}
    #Chars:     {len(en_chars)}

Hungarian Sentence lengths in Tokens:
    Max:        {hu_lens.max()}
    Min:        {hu_lens.min()}
    Mean:       {hu_lens.mean()}
    #Tokens:    {len(hu_toks)}
    #Chars:     {len(hu_chars)}

Difference in Sentence lengths in Tokens:
    Max:  {diff_lens.max()}
    Min:  {diff_lens.min()}
    Mean: {diff_lens.mean()}
"""
)

df["en_tokens"] = df["en_processed"].apply(wordpunct_tokenize)
df["hu_tokens"] = df["hu_processed"].apply(wordpunct_tokenize)
df = trim_outliers(df, "en_tokens", MAX_WORDS)
df = trim_outliers(df, "hu_tokens", MAX_WORDS)
df = remove_duplicates(df, "en_tokens")
df = remove_duplicates(df, "hu_tokens")

en_lens = get_sentence_lengths(df["en_processed"])
en_toks = get_list_of_unique_tokens(df["en_processed"])
en_chars = get_list_of_unique_chars(df["en_processed"])
hu_lens = get_sentence_lengths(df["hu_processed"])
hu_toks = get_list_of_unique_tokens(df["hu_processed"])
hu_chars = get_list_of_unique_chars(df["hu_processed"])
diff_lens = en_lens - hu_lens

plt.hist(en_lens, bins=10, alpha=0.45, color="red")
plt.hist(hu_lens, bins=10, alpha=0.45, color="blue")

plt.title(
    "Distribution of sentence lengths for each language\nafter limiting sentence length"
)

plt.legend(["English", "Hungarian"])

plt.show()

print(
    f"""
------------------------------------------------
AFTER LIMITING SENTENCE LENGTH TO {MAX_WORDS}
Dataset contains a total of {len(df)} entries.

English Sentence lengths in Tokens:
    Max:        {en_lens.max()}
    Min:        {en_lens.min()}
    Mean:       {en_lens.mean()}
    #Tokens:    {len(en_toks)}
    #Chars:     {len(en_chars)}

Hungarian Sentence lengths in Tokens:
    Max:        {hu_lens.max()}
    Min:        {hu_lens.min()}
    Mean:       {hu_lens.mean()}
    #Tokens:    {len(hu_toks)}
    #Chars:     {len(hu_chars)}

Difference in Sentence lengths in Tokens:
    Max:  {diff_lens.max()}
    Min:  {diff_lens.min()}
    Mean: {diff_lens.mean()}
------------------------------------------------
"""
)
