import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
import numpy as np
import torch
from torch.utils.data import Dataset

from data.preprocessing import sent2idx
from utils.constants import MAX_TOKENS


class LanguageDataset(Dataset):
    def __init__(
        self,
        input_sentences,
        output_sentences,
        input_word2idx,
        output_word2idx,
        max_tokens=MAX_TOKENS,
    ) -> None:
        self.input_sentences = input_sentences
        self.output_sentences = output_sentences
        self.input_word2idx = input_word2idx
        self.output_word2idx = output_word2idx
        self.max_tokens = max_tokens

    def __getitem__(self, index):
        data = sent2idx(
            self.input_word2idx, self.input_sentences[index], self.max_tokens
        )
        target = sent2idx(
            self.output_word2idx, self.output_sentences[index], self.max_tokens
        )
        return torch.LongTensor(data), torch.LongTensor(target)

    def __len__(self):
        return len(self.input_sentences)
