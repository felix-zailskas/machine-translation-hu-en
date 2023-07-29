import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from nltk.tokenize import wordpunct_tokenize

from data.preprocessing import add_sentence_tokens
from model.eval import evaluate_dataset
from model.model import AttnDecoderRNN, DecoderRNN, EncoderRNN, device
from utils.constants import *



learning_rates = [0.1, 0.01, 0.001]
batch_sizes = [32, 64, 128]

# Set Parameters before testing
in_lang = "en"
out_lang = "hu"
translation = f"{in_lang}_{out_lang}"
embedding = "cbow"
attention = "basic"
model_specific_name = "10_10"
dataset = "test"
topk = 3

df = pd.read_csv(f"data/preprocessed/preprocessed_val.csv")
df["en_processed"] = df["en_processed"].apply(wordpunct_tokenize)
df["hu_processed"] = df["hu_processed"].apply(wordpunct_tokenize)
df["en_processed"] = df["en_processed"].apply(add_sentence_tokens)
df["hu_processed"] = df["hu_processed"].apply(add_sentence_tokens)

input_sentences = df[f"{in_lang}_processed"]
output_sentences = df[f"{out_lang}_processed"]

input_pretrained_embeddings = torch.load(
    f"models/w2v_embeddings/embeddings_{embedding}_{in_lang}.pt"
)
output_pretrained_embeddings = torch.load(
    f"models/w2v_embeddings/embeddings_{embedding}_{out_lang}.pt"
)

with open(f"models/word2index/word2index_{embedding}_{in_lang}.pkl", "rb") as fp:
    input_word2idx = pickle.load(fp)
with open(f"models/word2index/word2index_{embedding}_{out_lang}.pkl", "rb") as fp:
    output_word2idx = pickle.load(fp)
with open(f"models/word2index/index2word_{embedding}_{in_lang}.pkl", "rb") as fp:
    input_idx2word = pickle.load(fp)
with open(f"models/word2index/index2word_{embedding}_{out_lang}.pkl", "rb") as fp:
    output_idx2word = pickle.load(fp)
    
 
encoder = EncoderRNN(
    len(input_word2idx), EMBEDDING_DIM, input_pretrained_embeddings
).to(device)
if attention == "basic":
    decoder = DecoderRNN(
        EMBEDDING_DIM, len(output_word2idx), output_pretrained_embeddings
    ).to(device)
elif attention == "attention":
    decoder = AttnDecoderRNN(
        EMBEDDING_DIM, len(output_word2idx), output_pretrained_embeddings
    ).to(device)   

one_gram_scores = {}
two_gram_scores = {}

for learning_rate in learning_rates:
    for batch_size in batch_sizes:
        encoder.load_state_dict(
            torch.load(
                f"models/tuning/encoder_{MAX_WORDS}_100_100_{str(learning_rate).replace('.','')}_{batch_size}.model"
            )
        )
        encoder.eval()
        decoder.load_state_dict(
            torch.load(
                f"models/tuning/decoder_{MAX_WORDS}_100_100_{str(learning_rate).replace('.','')}_{batch_size}.model"
            )
        )
        decoder.eval()
        one_gram_bleu_scores, predictions = evaluate_dataset(
            encoder,
            decoder,
            input_sentences,
            output_sentences,
            input_word2idx,
            output_idx2word,
            weights=(1, 0, 0, 0),
        )
        two_gram_bleu_scores, _ = evaluate_dataset(
            encoder,
            decoder,
            input_sentences,
            output_sentences,
            input_word2idx,
            output_idx2word,
            weights=(0.5, 0.5, 0, 0),
        )
        one_gram_scores[f"{str(learning_rate).replace('.','')}_{batch_size}"] = one_gram_bleu_scores.mean()
        two_gram_scores[f"{str(learning_rate).replace('.','')}_{batch_size}"] = two_gram_bleu_scores.mean()



ranks = {}
for i, (key, value) in enumerate(sorted(one_gram_scores.items(), key = lambda item: -item[1])):
    ranks[key] = i + 1
for i, (key, value) in enumerate(sorted(two_gram_scores.items(), key = lambda item: -item[1])):
    ranks[key] += i + 1
    ranks[key] /= 2

for key, value in sorted(ranks.items(), key = lambda item: item[1]):
    print(f"{key}: {value}")