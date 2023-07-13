import string

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Union, List, Callable
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from vectorize import get_n_gram, get_tfidf
import nltk
from nltk.corpus import stopwords  # note: download stopwords corpus
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from num2words import num2words

# global vars
# download stop words if not present on machine
try:
    stop_words = set(stopwords.words("english"))
except:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
wnl = WordNetLemmatizer()


## Normalisation methods used for every preprocessing pipeline ##
def make_lower(s: str) -> str:
    return str(s).lower()


def strip_whitespace(s: str) -> str:
    return s.strip()


def remove_punctuation(s: str) -> str:
    dic = {
        e: " "
        for e in string.punctuation.replace("#", "")  ## other symbols to include? *, !?
    }
    return str(s).translate(str.maketrans(dic))


def numbers_to_words(s: str) -> str:
    return " ".join([num2words(x) if x.isdigit() else x for x in s.split()])


def remove_stops(s: str) -> str:
    return " ".join([x for x in s.split() if x not in stop_words])


def remove_empty_text(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["processed"].str.len() > 0
    return_df = df.loc[mask]
    return return_df.reset_index(drop=True)


## Methods which differ in inclusion amongst different pipelines ##
def stem(s: str) -> list[str]:
    return " ".join([stemmer.stem(x) for x in s.split()])


def lemmatize(s: str) -> list[str]:
    return " ".join([wnl.lemmatize(x) for x in s.split()])


def preprocess_and_split(
    vec_mode: str,
    pipeline: List[Callable],
    sentiment_mapping: Dict[str, Union[int, np.ndarray]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    ngram_range: Tuple[int, int] = (1, 1),
    exclude_sentiment: List[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    assert train_ratio + val_ratio + test_ratio == 1.0
    df = pd.read_csv("../data/TweetSentiment.csv")
    df = df[["text", "sentiment"]]
    if exclude_sentiment is not None:
        for exclude_name in exclude_sentiment:
            df = df.drop(df[df["sentiment"] == exclude_name].index)
    df["sentiment"] = df["sentiment"].map(sentiment_mapping)
    df["text"] = df["text"].apply(str)

    df["processed"] = df["text"]
    for f in pipeline:
        df["processed"] = df["processed"].apply(f)
    df = remove_empty_text(df)

    if vec_mode == "tfidf":
        vectors, words = get_tfidf(df["processed"].to_list(), ngram_range=ngram_range)
    elif vec_mode == "ngram":
        vectors, words = get_n_gram(df["processed"].to_list(), ngram_range=ngram_range)
    vectors = vectors.toarray()
    df["vectors"] = None
    for (index, row), value in zip(df.iterrows(), vectors):
        df.at[index, "vectors"] = value

    X_raw = np.vstack(df["vectors"].to_numpy())
    Y_raw = np.vstack(df["sentiment"].to_numpy())
    _, counts = np.unique(Y_raw, return_counts=True)
    max_num = np.max(counts)
    oversampler = RandomOverSampler(
        sampling_strategy={i: max_num for i, _ in enumerate(counts)}
    )
    X_over, y_over = oversampler.fit_resample(X_raw, Y_raw)
    if y_over.shape[1] != len(sentiment_mapping):

        def make_vector(x, dim):
            vec = np.zeros(dim)
            vec[x] = 1
            return vec

        new_y = np.zeros((y_over.shape[0], len(sentiment_mapping)))
        for i, val in enumerate(y_over):
            new_y[i] = make_vector(val, len(sentiment_mapping))
        y_over = new_y
    X_train, X_test_val, Y_train, Y_test_val = train_test_split(
        X_over, y_over, test_size=1 - train_ratio, shuffle=True, stratify=y_over
    )

    X_val, X_test, Y_val, Y_test = train_test_split(
        X_test_val,
        Y_test_val,
        test_size=test_ratio / (test_ratio + val_ratio),
        shuffle=True,
        stratify=Y_test_val,
    )

    return X_train, X_val, X_test, Y_train, Y_val, Y_test, words
