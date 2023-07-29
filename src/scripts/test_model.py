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
from utils.constants import EMBEDDING_DIM

# Set Parameters before testing
in_lang = "en"
out_lang = "hu"
translation = f"{in_lang}_{out_lang}"
embedding = "cbow"
attention = "basic"
model_specific_name = "10_10"
dataset = "test"
topk = 3

df = pd.read_csv(f"data/preprocessed/preprocessed_{dataset}.csv")
df["en_processed"] = df["en_processed"].apply(wordpunct_tokenize)
df["hu_processed"] = df["hu_processed"].apply(wordpunct_tokenize)
df["en_processed"] = df["en_processed"].apply(add_sentence_tokens)
df["hu_processed"] = df["hu_processed"].apply(add_sentence_tokens)

input_sentences = df[f"{in_lang}_processed"]
output_sentences = df[f"{out_lang}_processed"]

print(
    f"Testing model models/{translation}/{embedding}/{attention}/encoder_{model_specific_name}.model on a total of {len(input_sentences)} sentences..."
)

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


# load trained model
encoder.load_state_dict(
    torch.load(
        f"models/{translation}/{embedding}/{attention}/encoder_{model_specific_name}.model"
    )
)
encoder.eval()
decoder.load_state_dict(
    torch.load(
        f"models/{translation}/{embedding}/{attention}/decoder_{model_specific_name}.model"
    )
)
decoder.eval()

one_gram_bleu_scores, predictions = evaluate_dataset(
    encoder,
    decoder,
    input_sentences,
    output_sentences,
    input_word2idx,
    output_word2idx,
    weights=(1, 0, 0, 0),
)
two_gram_bleu_scores, _ = evaluate_dataset(
    encoder,
    decoder,
    input_sentences,
    output_sentences,
    input_word2idx,
    output_word2idx,
    weights=(0.5, 0.5, 0, 0),
)

one_gram_order = np.argsort(one_gram_bleu_scores)
two_gram_order = np.argsort(two_gram_bleu_scores)


def display_topk_translations(
    input_sentences, output_sentences, target_senteces, scores, k, bottomk=False
):
    order = np.argsort(scores)
    for i in range(k):
        if bottomk:
            i *= -1
        idx = order[-i]
        print("Score: ", scores[idx])
        print("Input Sentece: ", " ".join(input_sentences[idx][1:-1]))
        print("Translation: ", " ".join(output_sentences[idx][1:-1]))
        print("Target Sentece: ", " ".join(target_senteces[idx][1:-1]))


def compute_score_per_sentece_length(input_lengths, scores):
    sentence_scores = {}
    for length, score in zip(input_lengths, scores):
        if length in sentence_scores:
            sentence_scores[length].append(score)
        else:
            sentence_scores[length] = [score]

    average_scores = {}
    for length, score_list in sentence_scores.items():
        average_scores[length] = np.mean(score_list)
    return average_scores


print(
    f"""
    Model Results:
        1-Gram bleu Score:
            Max: {one_gram_bleu_scores.max()}
            Min: {one_gram_bleu_scores.min()}
            Mean: {one_gram_bleu_scores.mean()}
        1-2-Gram bleu Score:
            Max: {two_gram_bleu_scores.max()}
            Min: {two_gram_bleu_scores.min()}
            Mean: {two_gram_bleu_scores.mean()}
"""
)

print("Best 1-gram translations:")
display_topk_translations(
    input_sentences, predictions, output_sentences, one_gram_bleu_scores, topk
)
print("Worst 1-gram translations:")
display_topk_translations(
    input_sentences,
    predictions,
    output_sentences,
    one_gram_bleu_scores,
    topk,
    bottomk=True,
)
print("Best 1-2-gram translations:")
display_topk_translations(
    input_sentences, predictions, output_sentences, two_gram_bleu_scores, topk
)
print("Worst 1-2-gram translations:")
display_topk_translations(
    input_sentences,
    predictions,
    output_sentences,
    two_gram_bleu_scores,
    topk,
    bottomk=True,
)

# Show Plots
plt.hist(one_gram_bleu_scores)
plt.title("Distribution of BLEU scores for 1-gram evaluation")
plt.xlabel("1-gram BLEU Score")
plt.ylabel("Frequency")
plt.show()

plt.hist(two_gram_bleu_scores)
plt.title("Distribution of BLEU scores for 1-gram and 2-gram average evaluation")
plt.xlabel("1-gram and 2-gram average BLEU Score")
plt.ylabel("Frequency")
plt.show()


input_lengths = np.array([len(sent) - 2 for sent in input_sentences])
one_gram_avg_scores = compute_score_per_sentece_length(
    input_lengths, one_gram_bleu_scores
)
two_gram_avg_scores = compute_score_per_sentece_length(
    input_lengths, two_gram_bleu_scores
)


plt.plot(
    list(one_gram_avg_scores.keys()),
    list(one_gram_avg_scores.values()),
    marker="o",
    linestyle="-",
)
plt.title("Average 1-gram BLEU score compared to input sentence length")
plt.xlabel("Sentence length")
plt.ylabel("BLEU Score")
plt.grid(True)
plt.show()

plt.plot(
    list(two_gram_avg_scores.keys()),
    list(two_gram_avg_scores.values()),
    marker="o",
    linestyle="-",
)
plt.title("Average 1-gram BLEU score compared to input sentence length")
plt.xlabel("Sentence length")
plt.ylabel("BLEU Score")
plt.grid(True)
plt.show()
