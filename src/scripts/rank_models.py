import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from pathlib import Path

from tqdm import tqdm

from model.eval import evaluate_dataset
from utils.config import load_dataset, load_models_from_config, read_config_file

word_ranks = {}
char_ranks = {}
word_one_gram_scores = {}
word_two_gram_scores = {}
char_one_gram_scores = {}
char_two_gram_scores = {}

for path in tqdm(Path("cfg").glob("*.yaml")):
    config = read_config_file(path)
    # Set Parameters before training
    in_lang = config["in_lang"]
    out_lang = config["out_lang"]
    translation = f"{in_lang}_{out_lang}"
    embedding = config["embedding"]
    attention = config["attention"]
    n_epochs = config["n_epochs"]
    dataset = "test"

    # word_input_sentences, word_output_sentences, word_max_tokens = load_dataset(in_lang=in_lang, out_lang=out_lang, dataset=dataset)
    char_input_sentences, char_output_sentences, char_max_tokens = load_dataset(
        in_lang=in_lang, out_lang=out_lang, dataset=dataset, char_model=True
    )

    # word_encoder, word_decoder, (word_input_word2idx, word_output_word2idx, word_input_idx2word, word_output_idx2word) = load_models_from_config(config, word_max_tokens)
    # word_encoder.eval()
    # word_decoder.eval()
    (
        char_encoder,
        char_decoder,
        (
            char_input_word2idx,
            char_output_word2idx,
            char_input_idx2word,
            char_output_idx2word,
        ),
    ) = load_models_from_config(config, char_max_tokens, char_model=True)
    char_encoder.eval()
    char_decoder.eval()

    # Rank models here
    # word_all_bleu_scores, word_predictions = evaluate_dataset(
    #     word_encoder,
    #     word_decoder,
    #     word_input_sentences,
    #     word_output_sentences,
    #     word_input_word2idx,
    #     word_output_idx2word,
    #     weights=[(1, 0, 0, 0), (0.5, 0.5, 0, 0)],
    # )
    char_all_bleu_scores, char_predictions = evaluate_dataset(
        char_encoder,
        char_decoder,
        char_input_sentences,
        char_output_sentences,
        char_input_word2idx,
        char_output_idx2word,
        weights=[(1, 0, 0, 0), (0.5, 0.5, 0, 0)],
        max_tokens=char_max_tokens,
        char_model=True,
    )
    # word_one_gram_bleu_scores = word_all_bleu_scores[0]
    # word_two_gram_bleu_scores = word_all_bleu_scores[1]
    # word_one_gram_scores[
    #     path
    # ] = word_one_gram_bleu_scores.mean()
    # word_two_gram_scores[
    #     path
    # ] = word_two_gram_bleu_scores.mean()
    char_one_gram_bleu_scores = char_all_bleu_scores[0]
    char_two_gram_bleu_scores = char_all_bleu_scores[1]
    char_one_gram_scores[path] = char_one_gram_bleu_scores.mean()
    char_two_gram_scores[path] = char_two_gram_bleu_scores.mean()

# for i, (key, value) in enumerate(
#     sorted(word_one_gram_scores.items(), key=lambda item: -item[1])
# ):
#     word_ranks[key] = [i + 1, [value]]
# for i, (key, value) in enumerate(
#     sorted(word_two_gram_scores.items(), key=lambda item: -item[1])
# ):
#     word_ranks[key][0] += i + 1
#     word_ranks[key][0] /= 2
#     word_ranks[key][1].append(value)

for i, (key, value) in enumerate(
    sorted(char_one_gram_scores.items(), key=lambda item: -item[1])
):
    char_ranks[key] = [i + 1, [value]]
for i, (key, value) in enumerate(
    sorted(char_two_gram_scores.items(), key=lambda item: -item[1])
):
    char_ranks[key][0] += i + 1
    char_ranks[key][0] /= 2
    char_ranks[key][1].append(value)

# print("WORD RANKING")
# for key, value in sorted(word_ranks.items(), key=lambda item: item[1]):
#     print(f"{key}: {value[0]}:: 1gram:{value[1][0]} 2gram:{value[1][1]}")
print("CHAR RANKING")
for key, value in sorted(char_ranks.items(), key=lambda item: item[1]):
    print(f"{key}: {value[0]}:: 1gram:{value[1][0]} 2gram:{value[1][1]}")
