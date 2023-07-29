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
from utils.constants import BATCH_SIZE, EMBEDDING_DIM, MAX_WORDS

# Set Parameters before training
in_lang = "en"
out_lang = "hu"
translation = f"{in_lang}_{out_lang}"
embedding = "cbow"
attention = "basic"
n_epochs = 10

df = pd.read_csv("data/preprocessed/preprocessed_train.csv")

df["en_processed"] = df["en_processed"].apply(wordpunct_tokenize)
df["hu_processed"] = df["hu_processed"].apply(wordpunct_tokenize)
df["en_processed"] = df["en_processed"].apply(add_sentence_tokens)
df["hu_processed"] = df["hu_processed"].apply(add_sentence_tokens)

input_sentences = df["en_processed"]
output_sentences = df["hu_processed"]

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

# create dataloader
dataset = LanguageDataset(
    input_sentences, output_sentences, input_word2idx, output_word2idx
)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# declare models
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

# train and save
train(dataloader, encoder, decoder, n_epochs, print_every=1, plot_every=1)
Path(f"models/{translation}/{embedding}/{attention}").mkdir(parents=True, exist_ok=True)
torch.save(
    encoder.state_dict(),
    f"models/{translation}/{embedding}/{attention}/encoder_{MAX_WORDS}_{n_epochs}.model",
)
torch.save(
    decoder.state_dict(),
    f"models/{translation}/{embedding}/{attention}/decoder_{MAX_WORDS}_{n_epochs}.model",
)

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
