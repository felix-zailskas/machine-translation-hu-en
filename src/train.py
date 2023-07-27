from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

import numpy as np
import pandas as pd

from lang import LanguageDataset
from preprocessing import get_vocab, word2idx, add_sentence_tokens
from nltk.tokenize import wordpunct_tokenize
from gensim.models import Word2Vec
from model import *

MAX_LENGTH = 140
embedding_dim = 100

df = pd.read_csv("data/preprocessed/preprocessed_val.csv")

df["en_processed"] = df["en_processed"].apply(wordpunct_tokenize)
df["hu_processed"] = df["hu_processed"].apply(wordpunct_tokenize)
df["en_processed"] = df["en_processed"].apply(add_sentence_tokens)
df["hu_processed"] = df["hu_processed"].apply(add_sentence_tokens)
input_sentences = df["en_processed"]
output_sentences = df["hu_processed"]

# declare embedding models 
input_w2v_model = Word2Vec(
    sentences=input_sentences, vector_size=embedding_dim, window=5, min_count=1, workers=4
)
input_w2v_model.save("models/word2vec_en.model")
output_w2v_model = Word2Vec(
    sentences=output_sentences, vector_size=embedding_dim, window=5, min_count=1, workers=4
)
output_w2v_model.save("models/word2vec_hu.model")

# get vocab and set up word -> idx dictionaries
input_vocab = get_vocab(input_sentences.values)
input_word2idx = word2idx(input_vocab)
output_vocab = get_vocab(output_sentences.values)
output_word2idx = word2idx(output_vocab)

# extract Word2Vec embeddings for english and hu tokens -- might not need this?
input_pretrained_embeddings = torch.zeros(len(input_vocab), input_w2v_model.vector_size)
output_pretrained_embeddings = torch.zeros(len(output_vocab), output_w2v_model.vector_size)

for word, index in input_word2idx.items():
    input_pretrained_embeddings[index] = torch.tensor(input_w2v_model.wv[word])
for word, index in output_word2idx.items():
    output_pretrained_embeddings[index] = torch.tensor(output_w2v_model.wv[word])

# create dataloader
dataset = LanguageDataset(
    input_sentences, output_sentences, input_word2idx, output_word2idx
)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# declare models and train
n_epochs = 3
encoder = EncoderRNN(MAX_LENGTH, embedding_dim, input_pretrained_embeddings).to(device)
decoder = DecoderRNN(embedding_dim, MAX_LENGTH, output_pretrained_embeddings).to(device)

#train(dataloader, encoder, decoder, n_epochs, print_every=1, plot_every=5)
# torch.save(encoder.state_dict(), "models/encoder.model")
# torch.save(decoder.state_dict(), "models/decoder.model")

## model eval ##
# load trained model
encoder.load_state_dict(torch.load("models/encoder.model"))
encoder.eval()
decoder.load_state_dict(torch.load("models/decoder.model"))
decoder.eval()

# evaluate some sentence pairs
output_words = evaluate(encoder, decoder, input_sentences[0], input_word2idx, output_word2idx, input_w2v_model, output_w2v_model)
output_sentence = " ".join(output_words)
print(input_sentences[0])
print(output_sentence)