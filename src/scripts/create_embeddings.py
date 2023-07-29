import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import pandas as pd
from nltk.tokenize import wordpunct_tokenize

from data.embeddings import create_w2v_embedding
from data.preprocessing import add_sentence_tokens, trim_outliers
from utils.constants import MAX_WORDS

print("Create embedding for complete corpus...")
df_train = pd.read_csv("data/preprocessed/preprocessed_train.csv")
df_val = pd.read_csv("data/preprocessed/preprocessed_val.csv")
df_test = pd.read_csv("data/preprocessed/preprocessed_test.csv")

df = pd.concat([df_train, df_val, df_test])

df["en_processed"] = df["en_processed"].apply(wordpunct_tokenize)
df["hu_processed"] = df["hu_processed"].apply(wordpunct_tokenize)
df["en_processed"] = df["en_processed"].apply(add_sentence_tokens)
df["hu_processed"] = df["hu_processed"].apply(add_sentence_tokens)


create_w2v_embedding(df["en_processed"], "cbow_en")
create_w2v_embedding(df["hu_processed"], "cbow_hu")
