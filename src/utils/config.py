import os.path
import pickle
import sys

import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
import yaml
from nltk.tokenize import wordpunct_tokenize

from data.preprocessing import add_sentence_tokens
from model.model import AttnDecoderRNN, DecoderRNN, EncoderRNN, device
from utils.constants import CHAR_EMBEDDING_DIM, MAX_TOKENS, WORD_EMBEDDING_DIM


def read_config_file(path):
    with open(path, "r") as fp:
        config = yaml.safe_load(fp)
    return config


def load_models_from_config(
    config, max_tokens, char_model=False, model_specific_name="10_100"
):
    in_lang = config["in_lang"]
    out_lang = config["out_lang"]
    translation = f"{in_lang}_{out_lang}"
    embedding = config["embedding"]
    attention = config["attention"]
    embedding_dim = CHAR_EMBEDDING_DIM if char_model else WORD_EMBEDDING_DIM

    if embedding == "basic":
        input_pretrained_embeddings = None
        output_pretrained_embeddings = None
    else:
        input_pretrained_embeddings = torch.load(
            f"models/w2v_embeddings/embeddings_{embedding}_{'char_' if char_model else ''}{in_lang}.pt"
        )
        output_pretrained_embeddings = torch.load(
            f"models/w2v_embeddings/embeddings_{embedding}_{'char_' if char_model else ''}{out_lang}.pt"
        )

    with open(
        f"models/word2index/word2index_{'char_' if char_model else ''}{in_lang}.pkl",
        "rb",
    ) as fp:
        input_word2idx = pickle.load(fp)
    with open(
        f"models/word2index/word2index_{'char_' if char_model else ''}{out_lang}.pkl",
        "rb",
    ) as fp:
        output_word2idx = pickle.load(fp)
    with open(
        f"models/word2index/index2word_{'char_' if char_model else ''}{in_lang}.pkl",
        "rb",
    ) as fp:
        input_idx2word = pickle.load(fp)
    with open(
        f"models/word2index/index2word_{'char_' if char_model else ''}{out_lang}.pkl",
        "rb",
    ) as fp:
        output_idx2word = pickle.load(fp)

    encoder = EncoderRNN(
        len(input_word2idx), embedding_dim, input_embeddings=input_pretrained_embeddings
    ).to(device)
    if attention == "basic":
        decoder = DecoderRNN(
            embedding_dim,
            len(output_word2idx),
            output_embeddings=output_pretrained_embeddings,
            max_tokens=max_tokens,
        ).to(device)
    elif attention == "attention":
        decoder = AttnDecoderRNN(
            embedding_dim,
            len(output_word2idx),
            output_embeddings=output_pretrained_embeddings,
            max_tokens=max_tokens,
        ).to(device)

        # load trained model
    encoder.load_state_dict(
        torch.load(
            f"models/{translation}/{embedding}/{attention}/{'char_' if char_model else ''}encoder_{model_specific_name}.model"
        )
    )
    decoder.load_state_dict(
        torch.load(
            f"models/{translation}/{embedding}/{attention}/{'char_' if char_model else ''}decoder_{model_specific_name}.model"
        )
    )

    return (
        encoder,
        decoder,
        (input_word2idx, output_word2idx, input_idx2word, output_idx2word),
    )


def load_dataset(in_lang, out_lang, dataset, char_model=False):
    df = pd.read_csv(f"data/preprocessed/preprocessed_{dataset}.csv")
    if char_model:
        df["en_processed"] = df["en_processed"].apply(lambda s: [*s])
        df["hu_processed"] = df["hu_processed"].apply(lambda s: [*s])
    else:
        df["en_processed"] = df["en_processed"].apply(wordpunct_tokenize)
        df["hu_processed"] = df["hu_processed"].apply(wordpunct_tokenize)
    df["en_processed"] = df["en_processed"].apply(add_sentence_tokens)
    df["hu_processed"] = df["hu_processed"].apply(add_sentence_tokens)

    input_sentences = df[f"{in_lang}_processed"]
    output_sentences = df[f"{out_lang}_processed"]

    in_max_tokens = input_sentences.apply(len).max()
    out_max_tokens = output_sentences.apply(len).max()
    max_tokens = max(in_max_tokens, out_max_tokens)

    return input_sentences, output_sentences, max_tokens
