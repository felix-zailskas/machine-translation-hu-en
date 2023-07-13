from itertools import chain
from typing import List, Tuple

import multidict as multidict
import numpy as np
from wordcloud import WordCloud


def get_mean_max_min_of_strings(text_list: List[str]) -> Tuple[float, float, float]:
    sentences = [x.split() for x in text_list]
    sentence_lens = np.array([len(x) for x in sentences])
    return sentence_lens.mean(), sentence_lens.max(), sentence_lens.min()


def get_list_of_words(text_list: List[str]) -> List[str]:
    text_list = [x.split() for x in text_list]
    text_list = chain.from_iterable(text_list)
    return list(text_list)


def get_common_words_of_strings(text_list: List[str]) -> Tuple[int, str, str]:
    all_words = np.array(get_list_of_words(text_list))
    unique, pos = np.unique(all_words, return_inverse=True)
    counts = np.bincount(pos)
    max_pos = counts.argmax()
    min_pos = counts.argmin()
    return (
        len(all_words),
        len(unique),
        unique[max_pos],
        counts[max_pos],
        unique[min_pos],
        counts[min_pos],
    )


def get_frequency_dict_for_text(sentence: str) -> multidict.MultiDict:
    full_terms_dict = multidict.MultiDict()
    tmp_dict = {}

    # making dict for counting frequencies
    for text in sentence.split():
        val = tmp_dict.get(text, 0)
        tmp_dict[text.lower()] = val + 1
    for key in tmp_dict:
        full_terms_dict.add(key, tmp_dict[key])
    return full_terms_dict


def create_freq_wordcloud(word_list: List[str]) -> WordCloud:
    return WordCloud(background_color="white").generate_from_frequencies(
        get_frequency_dict_for_text(" ".join(word_list))
    )
