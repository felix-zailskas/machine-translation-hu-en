import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from model.eval import evaluate_dataset, evaluate_single_sentence
from utils.config import load_dataset, load_models_from_config, read_config_file

# Set Parameters before testing
cfg = "cfg/cfg0.yaml"
config = read_config_file(cfg)
in_lang = config["in_lang"]
out_lang = config["out_lang"]
dataset = "test"
attention = config["attention"]
char_model = False
topk = 1

input_sentences, output_sentences, max_tokens = load_dataset(in_lang, out_lang, dataset)

(
    encoder,
    decoder,
    (input_word2idx, output_word2idx, input_idx2word, output_idx2word),
) = load_models_from_config(config, max_tokens, char_model=char_model)
encoder.eval()
decoder.eval()

all_gram_bleu_scores, predictions = evaluate_dataset(
    encoder,
    decoder,
    input_sentences,
    output_sentences,
    input_word2idx,
    output_idx2word,
    weights=[(1, 0, 0, 0), (0.5, 0.5, 0, 0)],
)
one_gram_bleu_scores = all_gram_bleu_scores[0]
two_gram_bleu_scores = all_gram_bleu_scores[1]

one_gram_order = np.argsort(one_gram_bleu_scores)
two_gram_order = np.argsort(two_gram_bleu_scores)


def display_topk_translations(
    input_sentences, output_sentences, target_senteces, scores, k, bottomk=False
):
    order = np.argsort(scores)
    for i in range(k):
        if bottomk:
            i *= -1
        else:
            i += 1
        idx = order[-i]
        print("######")
        print("\tScore: ", scores[idx])
        print("\tInput Sentece: ", " ".join(input_sentences[idx][1:-1]))
        print("\tTranslation: ", " ".join(output_sentences[idx][1:-1]))
        print("\tTarget Sentece: ", " ".join(target_senteces[idx][1:-1]))


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
--------------------------------------------------------------
Model Results:
    1-Gram bleu Score:
        Max: {one_gram_bleu_scores.max()}
        Min: {one_gram_bleu_scores.min()}
        Mean: {one_gram_bleu_scores.mean()}
    1-2-Gram bleu Score:
        Max: {two_gram_bleu_scores.max()}
        Min: {two_gram_bleu_scores.min()}
        Mean: {two_gram_bleu_scores.mean()}
--------------------------------------------------------------
"""
)

print("Best 1-gram translations:")
display_topk_translations(
    input_sentences, predictions, output_sentences, one_gram_bleu_scores, topk
)
print("--------------------------------------------------------------")
print("Worst 1-gram translations:")
display_topk_translations(
    input_sentences,
    predictions,
    output_sentences,
    one_gram_bleu_scores,
    topk,
    bottomk=True,
)
print("--------------------------------------------------------------")
print("Best 1-2-gram translations:")
display_topk_translations(
    input_sentences, predictions, output_sentences, two_gram_bleu_scores, topk
)
print("--------------------------------------------------------------")
print("Worst 1-2-gram translations:")
display_topk_translations(
    input_sentences,
    predictions,
    output_sentences,
    two_gram_bleu_scores,
    topk,
    bottomk=True,
)
print("--------------------------------------------------------------")
print("Sentence Length Counts:")
input_lengths = np.array([len(sent) - 2 for sent in input_sentences])
one_gram_avg_scores = compute_score_per_sentece_length(
    input_lengths, one_gram_bleu_scores
)
two_gram_avg_scores = compute_score_per_sentece_length(
    input_lengths, two_gram_bleu_scores
)
unique_lengths, frequencies = np.unique(input_lengths, return_counts=True)
for length, frequency in zip(unique_lengths, frequencies):
    print(f"\tLength {length}: {frequency} sentences")
print("--------------------------------------------------------------")

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


x = list(one_gram_avg_scores.keys())
y = list(one_gram_avg_scores.values())
x, y = zip(*sorted(zip(x, y)))

plt.plot(x, y, marker="o")
plt.title("Average 1-gram BLEU score compared to input sentence length")
plt.xlabel("Sentence length")
plt.ylabel("BLEU Score")
plt.grid(True)
plt.show()

x = list(two_gram_avg_scores.keys())
y = list(two_gram_avg_scores.values())
x, y = zip(*sorted(zip(x, y)))

plt.plot(x, y, marker="o")
plt.title("Average 1-gram and 2-gram BLEU score compared to input sentence length")
plt.xlabel("Sentence length")
plt.ylabel("BLEU Score")
plt.grid(True)
plt.show()


def showAttention(input_sentence, output_words, attentions):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.cpu().numpy(), cmap="bone")
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([""] + input_sentence, rotation=90)
    ax.set_yticklabels([""] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate_single_sentence(
        encoder, decoder, input_sentence, input_word2idx, output_idx2word
    )
    print("input =", input_sentence)
    print("output =", " ".join(output_words))
    showAttention(input_sentence, output_words, attentions[0, : len(output_words), :])


if attention == "attention":
    evaluateAndShowAttention(input_sentences[np.argsort(one_gram_bleu_scores)[-1]])
    evaluateAndShowAttention(input_sentences[np.argsort(two_gram_bleu_scores)[-1]])
