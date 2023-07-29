import os.path
import pickle
import sys

import pandas as pd
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from pathlib import Path

import torch
from nltk.tokenize import wordpunct_tokenize

from data.dataset import LanguageDataset
from data.preprocessing import add_sentence_tokens
from model.model import AttnDecoderRNN, DecoderRNN, EncoderRNN, device
from model.train import train
from utils.constants import BATCH_SIZE, LEARNING_RATE, MAX_WORDS, WORD_EMBEDDING_DIM

# Set Parameters before training
in_lang = "en"
out_lang = "hu"
translation = f"{in_lang}_{out_lang}"
embedding = "cbow"
attention = "basic"
n_epochs = 1

df = pd.read_csv("data/preprocessed/preprocessed_train.csv")

df["en_processed"] = df["en_processed"].apply(wordpunct_tokenize)
df["hu_processed"] = df["hu_processed"].apply(wordpunct_tokenize)
df["en_processed"] = df["en_processed"].apply(add_sentence_tokens)
df["hu_processed"] = df["hu_processed"].apply(add_sentence_tokens)

input_sentences = df[f"{in_lang}_processed"]
output_sentences = df[f"{out_lang}_processed"]

if embedding == "basic":
    input_pretrained_embeddings = None
    output_pretrained_embeddings = None
else:
    input_pretrained_embeddings = torch.load(
        f"models/w2v_embeddings/embeddings_{embedding}_{in_lang}.pt"
    )
    output_pretrained_embeddings = torch.load(
        f"models/w2v_embeddings/embeddings_{embedding}_{out_lang}.pt"
    )

with open(f"models/word2index/word2index_{in_lang}.pkl", "rb") as fp:
    input_word2idx = pickle.load(fp)
with open(f"models/word2index/word2index_{out_lang}.pkl", "rb") as fp:
    output_word2idx = pickle.load(fp)
with open(f"models/word2index/index2word_{in_lang}.pkl", "rb") as fp:
    input_idx2word = pickle.load(fp)
with open(f"models/word2index/index2word_{out_lang}.pkl", "rb") as fp:
    output_idx2word = pickle.load(fp)

# create dataloader
dataset = LanguageDataset(
    input_sentences, output_sentences, input_word2idx, output_word2idx
)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# declare models
encoder = EncoderRNN(
    len(input_word2idx), WORD_EMBEDDING_DIM, input_pretrained_embeddings
).to(device)
if attention == "basic":
    decoder = DecoderRNN(
        WORD_EMBEDDING_DIM,
        len(output_word2idx),
        output_embeddings=output_pretrained_embeddings,
    ).to(device)
elif attention == "attention":
    decoder = AttnDecoderRNN(
        WORD_EMBEDDING_DIM,
        len(output_word2idx),
        output_embeddings=output_pretrained_embeddings,
    ).to(device)

# train and save
train(
    dataloader,
    encoder,
    decoder,
    n_epochs,
    learning_rate=LEARNING_RATE,
    print_every=1,
    plot_every=1,
)
Path(f"models/{translation}/{embedding}/{attention}").mkdir(parents=True, exist_ok=True)
torch.save(
    encoder.state_dict(),
    f"models/{translation}/{embedding}/{attention}/encoder_{MAX_WORDS}_{n_epochs}.model",
)
torch.save(
    decoder.state_dict(),
    f"models/{translation}/{embedding}/{attention}/decoder_{MAX_WORDS}_{n_epochs}.model",
)
