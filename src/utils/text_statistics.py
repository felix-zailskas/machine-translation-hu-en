from itertools import chain
from typing import List, Tuple

import multidict as multidict
import numpy as np
from wordcloud import WordCloud
from nltk.tokenize import wordpunct_tokenize


def get_sentence_lengths(text_list: List[str]):
    sentences = [wordpunct_tokenize(x) for x in text_list]
    sentence_lens = np.array([len(x) for x in sentences])
    return sentence_lens


def get_list_of_unique_tokens(text_list: List[str]) -> List[str]:
    text_list = [wordpunct_tokenize(x) for x in text_list]
    text_list = list(chain.from_iterable(text_list))
    return np.unique(np.array(text_list))


def get_list_of_unique_chars(char_list: List[str]) -> List[str]:
    char_list = [[*x] for x in char_list]
    char_list = list(chain.from_iterable(char_list))
    return np.unique(np.array(char_list))
