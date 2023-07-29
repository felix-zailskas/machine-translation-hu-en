import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
import numpy as np
import torch
from nltk.translate import bleu_score

from data.preprocessing import sent2idx
from utils.constants import EOS_TOKEN, MAX_TOKENS


def evaluate_single_sentence(
    encoder, decoder, sentence, input_word2idx, output_index2word, max_tokens=MAX_TOKENS
):
    with torch.no_grad():
        input_tensor = sent2idx(input_word2idx, sentence, max_tokens)
        input_tensor = torch.LongTensor(input_tensor)
        encoder_outputs, encoder_hidden = encoder(input_tensor)

        encoder_outputs = encoder_outputs.unsqueeze(0)
        encoder_hidden = encoder_hidden.unsqueeze(0)

        decoder_outputs, decoder_hidden, decoder_attn = decoder(
            encoder_outputs, encoder_hidden
        )
        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_TOKEN:
                decoded_words.append("<EOS>")
                break
            decoded_words.append(output_index2word[idx.item()])

    return decoded_words


def evaluate_dataset(
    encoder,
    decoder,
    input_sentences,
    expected_output,
    input_word2idx,
    output_idx2word,
    weights=(1, 0, 0, 0),
):
    predictions = [
        evaluate_single_sentence(
            encoder, decoder, test_input_sentence, input_word2idx, output_idx2word
        )[1:-1]
        for test_input_sentence in input_sentences
    ]
    targets = [sentence[1:-1] for sentence in expected_output]

    bleu_scores = []
    for prediction, target in zip(predictions, targets):
        bleu_scores.append(
            bleu_score.sentence_bleu([target], prediction, weights=weights)
        )

    bleu_scores = np.array(bleu_scores)
    return bleu_scores, predictions
