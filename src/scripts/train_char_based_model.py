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
from model.eval import evaluate_single_sentence
from model.model import AttnDecoderRNN, DecoderRNN, EncoderRNN, device
from model.train import train
from utils.config import read_config_file
from utils.constants import CHAR_EMBEDDING_DIM, MAX_WORDS

for path in Path("cfg").glob("*.yaml"):
    config = read_config_file(path)
    # Set Parameters before training
    in_lang = config["in_lang"]
    out_lang = config["out_lang"]
    translation = f"{in_lang}_{out_lang}"
    embedding = config["embedding"]
    attention = config["attention"]
    n_epochs = config["n_epochs"]
    lr = config["lr"]
    batch_size = config["batch_size"]

    df = pd.read_csv("data/preprocessed/preprocessed_train.csv")
    df["en_processed"] = df["en_processed"].apply(lambda s: [*s])
    df["hu_processed"] = df["hu_processed"].apply(lambda s: [*s])
    df["en_processed"] = df["en_processed"].apply(add_sentence_tokens)
    df["hu_processed"] = df["hu_processed"].apply(add_sentence_tokens)

    input_sentences = df[f"{in_lang}_processed"]
    output_sentences = df[f"{out_lang}_processed"]

    in_max_tokens = input_sentences.apply(len).max()
    out_max_tokens = output_sentences.apply(len).max()
    max_tokens = max(in_max_tokens, out_max_tokens)

    if embedding == "basic":
        input_pretrained_embeddings = None
        output_pretrained_embeddings = None
    else:
        input_pretrained_embeddings = torch.load(
            f"models/w2v_embeddings/embeddings_{embedding}_char_{in_lang}.pt"
        )
        output_pretrained_embeddings = torch.load(
            f"models/w2v_embeddings/embeddings_{embedding}_char_{out_lang}.pt"
        )

    with open(f"models/word2index/word2index_char_{in_lang}.pkl", "rb") as fp:
        input_word2idx = pickle.load(fp)
    with open(f"models/word2index/word2index_char_{out_lang}.pkl", "rb") as fp:
        output_word2idx = pickle.load(fp)
    with open(f"models/word2index/index2word_char_{in_lang}.pkl", "rb") as fp:
        input_idx2word = pickle.load(fp)
    with open(f"models/word2index/index2word_char_{out_lang}.pkl", "rb") as fp:
        output_idx2word = pickle.load(fp)

    # create dataloader
    dataset = LanguageDataset(
        input_sentences,
        output_sentences,
        input_word2idx,
        output_word2idx,
        max_tokens=max_tokens,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # declare models
    encoder = EncoderRNN(
        len(input_word2idx), CHAR_EMBEDDING_DIM, input_pretrained_embeddings
    ).to(device)
    if attention == "basic":
        decoder = DecoderRNN(
            CHAR_EMBEDDING_DIM,
            len(output_word2idx),
            output_embeddings=output_pretrained_embeddings,
            max_tokens=max_tokens,
        ).to(device)
    elif attention == "attention":
        decoder = AttnDecoderRNN(
            CHAR_EMBEDDING_DIM,
            len(output_word2idx),
            output_embeddings=output_pretrained_embeddings,
            max_tokens=max_tokens,
        ).to(device)

    # train and save
    train(
        dataloader,
        encoder,
        decoder,
        n_epochs,
        learning_rate=lr,
        print_every=1,
        plot_every=1,
    )
    Path(f"models/{translation}/{embedding}/{attention}").mkdir(
        parents=True, exist_ok=True
    )
    torch.save(
        encoder.state_dict(),
        f"models/{translation}/{embedding}/{attention}/char_encoder_{MAX_WORDS}_{n_epochs}.model",
    )
    torch.save(
        decoder.state_dict(),
        f"models/{translation}/{embedding}/{attention}/char_decoder_{MAX_WORDS}_{n_epochs}.model",
    )
