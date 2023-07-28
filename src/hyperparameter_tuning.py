import pandas as pd
import numpy as np
from model import EncoderRNN, DecoderRNN, train, device
from lang import LanguageDataset
from torch.utils.data import DataLoader
import torch
import pickle
from preprocessing import add_sentence_tokens
from nltk.tokenize import wordpunct_tokenize
from constants import *


embedding_sizes = [100, 200, 300]
learning_rates = [0.1, 0.01, 0.001]
batch_sizes = [32, 64, 128]
n_epochs = 100

translation = "en_hu"
embedding = "cbow"
attention = "basic"

df = pd.read_csv("data/preprocessed/preprocessed_train.csv")

df["en_processed"] = df["en_processed"].apply(wordpunct_tokenize)
df["hu_processed"] = df["hu_processed"].apply(wordpunct_tokenize)
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
    
dataset = LanguageDataset(
    input_sentences, output_sentences, input_word2idx, output_word2idx
)

for embedding_size in embedding_sizes:
    for learning_rate in learning_rates:
        for batch_size in batch_sizes:
            name = f"{embedding_size}_{str(learning_rate).replace('.','')}_{batch_size}"
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            encoder = EncoderRNN(len(input_word2idx), embedding_size, input_pretrained_embeddings).to(device)
            decoder = DecoderRNN(embedding_size, len(output_word2idx), output_pretrained_embeddings).to(device)
            train(dataloader, encoder, decoder, n_epochs, learning_rate=learning_rate, print_every=1, plot_every=1)
            torch.save(encoder.state_dict(), f"models/tuning/encoder_{MAX_WORDS}_{n_epochs}_{name}.model")
            torch.save(decoder.state_dict(), f"models/tuning/decoder_{MAX_WORDS}_{n_epochs}_{name}.model")
            
