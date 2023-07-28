from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random
from time import time
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

import numpy as np
import pandas as pd
import pickle

from lang import LanguageDataset
from preprocessing import get_vocab, word2idx, idx2word, add_sentence_tokens, trim_outliers
from nltk.tokenize import wordpunct_tokenize
from gensim.models import Word2Vec
from model import *
from constants import *

df = pd.read_csv("data/preprocessed/preprocessed_train.csv")

df["en_processed"] = df["en_processed"].apply(wordpunct_tokenize)
df["hu_processed"] = df["hu_processed"].apply(wordpunct_tokenize)
df = trim_outliers(df, "en_processed", MAX_WORDS)
df = trim_outliers(df, "hu_processed", MAX_WORDS)
print(len(df.index))
df["en_processed"] = df["en_processed"].apply(add_sentence_tokens)
df["hu_processed"] = df["hu_processed"].apply(add_sentence_tokens)

input_sentences = df["en_processed"]
output_sentences = df["hu_processed"]

input_pretrained_embeddings = torch.load("models/w2v_embeddings/embeddings_cbow_en.pt")
output_pretrained_embeddings = torch.load("models/w2v_embeddings/embeddings_cbow_hu.pt")

with open("models/word2index/word2index_cbow_en.pkl", 'rb') as fp:
    input_word2idx = pickle.load(fp)
with open("models/word2index/word2index_cbow_hu.pkl", 'rb') as fp:
    output_word2idx = pickle.load(fp)
with open("models/word2index/index2word_cbow_en.pkl", 'rb') as fp:
    input_idx2word = pickle.load(fp)
with open("models/word2index/index2word_cbow_hu.pkl", 'rb') as fp:
    output_idx2word = pickle.load(fp)

# create dataloader
dataset = LanguageDataset(
    input_sentences, output_sentences, input_word2idx, output_word2idx
)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# declare models 
encoder = EncoderRNN(len(input_word2idx), EMBEDDING_DIM, input_pretrained_embeddings).to(device)
#decoder = DecoderRNN(EMBEDDING_DIM, len(output_word2idx), output_pretrained_embeddings).to(device)
decoder = AttnDecoderRNN(EMBEDDING_DIM, len(output_word2idx), output_pretrained_embeddings).to(device)

# train and save ##
n_epochs = 100
train(dataloader, encoder, decoder, n_epochs, print_every=1, plot_every=1)
torch.save(encoder.state_dict(), f"models/encoder_attention_train_{MAX_WORDS}_{n_epochs}.model")
torch.save(decoder.state_dict(), f"models/decoder_attention_train_{MAX_WORDS}_{n_epochs}.model")

# # ## model eval ##
# # load trained model
# encoder.load_state_dict(torch.load("models/encoder.model"))
# encoder.eval()
# decoder.load_state_dict(torch.load("models/decoder.model"))
# decoder.eval()

# # evaluate some sentence pairs
# test_input_sentence = input_sentences[0]
# test_output_sentence = output_sentences[0]
# output_words = evaluate(encoder, decoder, test_input_sentence, input_word2idx, output_idx2word)
# output_sentence = " ".join(output_words)
# print(test_input_sentence)
# print("-----")
# print("Model output:", output_sentence)
# print("Target:", test_output_sentence)