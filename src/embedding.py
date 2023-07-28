from gensim.models import Word2Vec
from constants import *
import torch
from preprocessing import get_vocab
import pickle
import pandas as pd
from preprocessing import trim_outliers, add_sentence_tokens, word2idx, idx2word
from nltk.tokenize import wordpunct_tokenize


def create_w2v_embedding(sentences, name): 
    w2v_model = Word2Vec(
        sentences=sentences, vector_size=EMBEDDING_DIM, window=5, min_count=1, workers=4
    )
    vocab = get_vocab(sentences.values)
    w2i = word2idx(vocab)
    with open(f'models/word2index/word2index_{name}.pkl', 'wb') as fp:
        pickle.dump(w2i, fp)
    i2w = idx2word(vocab)
    with open(f'models/word2index/index2word_{name}.pkl', 'wb') as fp:
        pickle.dump(i2w, fp)
    pretrained_embeddings = torch.zeros(len(vocab), w2v_model.vector_size)
    for word, index in w2i.items():
        pretrained_embeddings[index] = torch.tensor(w2v_model.wv[word])
    torch.save(pretrained_embeddings, f'models/w2v_embeddings/embeddings_{name}.pt')
    
    
df_train = pd.read_csv("data/preprocessed/preprocessed_train.csv")
df_val = pd.read_csv("data/preprocessed/preprocessed_val.csv")
df_test = pd.read_csv("data/preprocessed/preprocessed_test.csv")

df = pd.concat([df_train, df_val, df_test])

df["en_processed"] = df["en_processed"].apply(wordpunct_tokenize)
df["hu_processed"] = df["hu_processed"].apply(wordpunct_tokenize)
df = trim_outliers(df, "en_processed", MAX_WORDS)
df = trim_outliers(df, "hu_processed", MAX_WORDS)
print(len(df.index))
df["en_processed"] = df["en_processed"].apply(add_sentence_tokens)
df["hu_processed"] = df["hu_processed"].apply(add_sentence_tokens)

create_w2v_embedding(df["en_processed"], "cbow_en")
create_w2v_embedding(df["hu_processed"], "cbow_hu")