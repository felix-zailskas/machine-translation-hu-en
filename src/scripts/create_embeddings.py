import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import pandas as pd
from nltk.tokenize import wordpunct_tokenize

from data.embeddings import create_w2v_embedding, create_word2index_tables
from data.preprocessing import add_sentence_tokens, trim_outliers
from utils.constants import CHAR_EMBEDDING_DIM, MAX_WORDS

print("Create embedding for complete corpus...")
df_train = pd.read_csv("data/preprocessed/preprocessed_train.csv")
df_val = pd.read_csv("data/preprocessed/preprocessed_val.csv")
df_test = pd.read_csv("data/preprocessed/preprocessed_test.csv")

df = pd.concat([df_train, df_val, df_test])

en_all_text = " ".join(df["en_processed"].to_list())
en_chars = list(set(en_all_text)) + [["<SOS>"], ["<EOS>"]]
hu_all_text = " ".join(df["hu_processed"].to_list())
hu_chars = list(set(hu_all_text)) + [["<SOS>"], ["<EOS>"]]

df["en_processed"] = df["en_processed"].apply(wordpunct_tokenize)
df["hu_processed"] = df["hu_processed"].apply(wordpunct_tokenize)
df["en_processed"] = df["en_processed"].apply(add_sentence_tokens)
df["hu_processed"] = df["hu_processed"].apply(add_sentence_tokens)

# Word based models
w2i_en, _ = create_word2index_tables(df["en_processed"], "en")
w2i_hu, _ = create_word2index_tables(df["hu_processed"], "hu")
create_w2v_embedding(df["en_processed"], "cbow_en", w2i=w2i_en)
create_w2v_embedding(df["hu_processed"], "cbow_hu", w2i=w2i_hu)
create_w2v_embedding(df["en_processed"], "skip_en", w2i=w2i_en, sg=1)
create_w2v_embedding(df["hu_processed"], "skip_hu", w2i=w2i_hu, sg=1)

# Character based models
c2i_en, _ = create_word2index_tables(pd.Series(en_chars), "char_en")
c2i_hu, _ = create_word2index_tables(pd.Series(hu_chars), "char_hu")
create_w2v_embedding(
    pd.Series(en_chars), "cbow_char_en", w2i=c2i_en, embedding_dim=CHAR_EMBEDDING_DIM
)
create_w2v_embedding(
    pd.Series(hu_chars), "cbow_char_hu", w2i=c2i_hu, embedding_dim=CHAR_EMBEDDING_DIM
)
