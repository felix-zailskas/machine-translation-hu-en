import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
import pickle
from pathlib import Path

import torch
from gensim.models import Word2Vec

from data.preprocessing import get_vocab, idx2word, word2idx
from utils.constants import EMBEDDING_DIM


def create_w2v_embedding(sentences, name, embdedding_dim=EMBEDDING_DIM):
    w2v_model = Word2Vec(
        sentences=sentences,
        vector_size=embdedding_dim,
        window=5,
        min_count=1,
        workers=4,
    )
    Path("models/word2index").mkdir(parents=True, exist_ok=True)
    Path("models/w2v_embeddings").mkdir(parents=True, exist_ok=True)
    vocab = get_vocab(sentences.values)
    w2i = word2idx(vocab)
    with open(f"models/word2index/word2index_{name}.pkl", "wb") as fp:
        pickle.dump(w2i, fp)
    i2w = idx2word(vocab)
    with open(f"models/word2index/index2word_{name}.pkl", "wb") as fp:
        pickle.dump(i2w, fp)
    pretrained_embeddings = torch.zeros(len(vocab), w2v_model.vector_size)
    for word, index in w2i.items():
        pretrained_embeddings[index] = torch.tensor(w2v_model.wv[word])
    torch.save(pretrained_embeddings, f"models/w2v_embeddings/embeddings_{name}.pt")
