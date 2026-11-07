import math
import os
import random
import re
import sys
from typing import Self

import numpy as np

try:
    from gensim.models import KeyedVectors
except ImportError:
    KeyedVectors = None


if sys.version_info < (3, 8):
    sys.exit("This program requires Python 3.8 or newer.")


TAGS = [
    "research", "innovation", "education", "university", "students",
    "faculty", "campus", "engineering", "medicine", "technology",
    "curriculum", "collaboration", "publication", "laboratory",
    "scholarship", "mentorship", "internship", "entrepreneurship",
    "accreditation", "alumni"
]
# The list includes the basic function words and a few words noticed in
# The Linkedin text is used for this assignment.
STOPWORDS = { "a", "an", "the", "and", "but", "or", "in", "on", "at", "to", "for","from", "with", "by", "of", 
            "is", "am", "are", "was", "were", "be","i", "you", "he", "she", "it", "we", "they", "this", "that",
              "our","their", "has", "have", "as", "can", "not", "its", "us", "who","will", "more", "than", 
              "into", "through", "over", "across"}

def load_model(word2vec_model_path: str):
    if KeyedVectors is None:
        raise ImportError("Install gensim please to continue")
    model_path = os.path.expanduser(word2vec_model_path)
    return KeyedVectors.load(model_path, mmap="r")

class Vec:
    def __init__(self, model_or_elements=None, word=None):
        if word is not None:
            if not isinstance(word, str):
                raise TypeError("The word argument must be a string.")
            try:
                values = model_or_elements[word]
            except KeyError as error:
                raise ValueError(f"Word '{word}' is not in the model vocabulary.") from error
        else:
            values = model_or_elements

        if values is None:
            self.elements = []
        else:
            self.elements = []
            for value in values:
                if isinstance(value, (bool, np.bool_)) or not isinstance(
                    value, (int, float, np.integer, np.floating)
                ):
                    raise TypeError("Every vector element must be a number.")
                self.elements.append(float(value))

    def __len__(self):
        return len(self.elements)

    def __repr__(self):
        return repr(self.elements)

    def _check_vector(self, other):
        if not isinstance(other, Vec):
            raise TypeError("The other operand must be a Vec.")
        if len(self) != len(other):
            raise ValueError("Vectors must have the same dimensions.")

    def __add__(self, other: Self) -> Self:
        self._check_vector(other)
        return Vec([a + b for a, b in zip(self.elements, other.elements)])

    def __sub__(self, other: Self) -> Self:
        self._check_vector(other)
        return Vec([a - b for a, b in zip(self.elements, other.elements)])

    def __neg__(self) -> Self:
        return Vec([-value for value in self.elements])

    def __rmul__(self, scalar: int | float) -> Self:
        if isinstance(scalar, bool) or not isinstance(scalar, (int, float, np.number)):
            raise TypeError("The multiplier must be a number.")
        return Vec([value * scalar for value in self.elements])

    def __mul__(self, other):
        if isinstance(other, Vec):
            self._check_vector(other)
            return Vec([a * b for a, b in zip(self.elements, other.elements)])
        if isinstance(other, bool) or not isinstance(other, (int, float, np.number)):
            raise TypeError("The multiplier must be a number or another Vec.")
        return Vec([value * other for value in self.elements])

    def __imul__(self, scalar: int | float) -> Self:
        if isinstance(scalar, bool) or not isinstance(scalar, (int, float, np.number)):
            raise TypeError("The multiplier must be a number.")
        self.elements = [value * scalar for value in self.elements]
        return self

    def __iadd__(self, other: Self) -> Self:
        self._check_vector(other)
        self.elements = [
            a + b for a, b in zip(self.elements, other.elements)
        ]
        return self

    def vec_mean(self, src=None) -> float:
        if src is not None:
            self.elements = list(src)
        if not self.elements:
            raise ValueError("Cannot calculate the mean of an empty vector.")
        return sum(self.elements) / len(self.elements)

    def vec_demean(self, src=None) -> Self:
        if src is not None:
            self.elements = list(src)
        mean_value = self.vec_mean()
        self.elements = [value - mean_value for value in self.elements]
        return self

    def vec_std(self, src=None) -> float:
        if src is not None:
            self.elements = list(src)
        if not self.elements:
            raise ValueError("Cannot calculate standard deviation of an empty vector.")
        mean_value = self.vec_mean()
        squared_diff = [(value - mean_value) ** 2 for value in self.elements]
        return math.sqrt(sum(squared_diff) / len(squared_diff))

    @staticmethod
    def zeros(n: int) -> Self:
        if not isinstance(n, int) or isinstance(n, bool):
            raise TypeError("The vector length must be an integer.")
        if n <= 0:
            raise ValueError("The vector length must be greater than zero.")
        return Vec([0] * n)

    @staticmethod
    def ones(n: int) -> Self:
        if not isinstance(n, int) or isinstance(n, bool):
            raise TypeError("The vector length must be an integer.")
        if n <= 0:
            raise ValueError("The vector length must be greater than zero.")
        return Vec([1] * n)

    @staticmethod
    def uniform(n: int) -> Self:
        if not isinstance(n, int) or isinstance(n, bool):
            raise TypeError("The vector length must be an integer.")
        if n <= 0:
            raise ValueError("The vector length must be greater than zero.")
        return Vec([random.uniform(0, 1) for _ in range(n)])

    def norm(self) -> float:
        total = sum(value * value for value in self.elements)
        return math.sqrt(total)

    def dot(self, other: Self) -> float:
        self._check_vector(other)
        return sum(a * b for a, b in zip(self.elements, other.elements))

    def cosine_similarity(self, other: Self) -> float:
        self._check_vector(other)
        first_norm = self.norm()
        second_norm = other.norm()
        if first_norm == 0 or second_norm == 0:
            raise ValueError("Cosine similarity is undefined for a zero vector.")
        return self.dot(other) / (first_norm * second_norm)

    @staticmethod
    def preprocess(text: str) -> list:
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = text.replace("_", " ")
        tokens = text.split()
        tokens = [word for word in tokens if word not in STOPWORDS]
        return [word for word in tokens if len(word) > 2]

    @staticmethod
    def load_model(model_path: str):
        return load_model(model_path)


def build_tag_matrix(model, tags):
    if len(tags) != 20:
        raise ValueError(f"Exactly 20 tags are required, got {len(tags)}.")

    rows = []
    for tag in tags:
        words = tag.split()
        if not words:
            raise ValueError("A tag cannot be empty.")

        for word in words:
            if word not in model.key_to_index:
                raise ValueError(f"Tag not in vocabulary: {word}")

        word_vectors = np.asarray(
            [model[word] for word in words], dtype=np.float64
        )
        rows.append(np.mean(word_vectors, axis=0))

    return list(tags), np.vstack(rows)


def build_text_matrix(model, tokens):
    in_vocab_tokens = []
    oov_tokens = []

    for token in tokens:
        if token in model.key_to_index:
            in_vocab_tokens.append(token)
        else:
            oov_tokens.append(token)

    if not in_vocab_tokens:
        raise ValueError("No processed words were found in the model vocabulary.")

    W = np.asarray(
        [model[token] for token in in_vocab_tokens], dtype=np.float64
    )
    return in_vocab_tokens, oov_tokens, W


def similarity_matrix_vector_class(W, T):
    result = []
    tag_vectors = [Vec(row) for row in T]

    for word_row in W:
        word_vector = Vec(word_row)
        similarities_for_word = []

        for tag_vector in tag_vectors:
            similarities_for_word.append(
                word_vector.cosine_similarity(tag_vector)
            )

        result.append(similarities_for_word)

    return np.asarray(result, dtype=np.float64)


def similarity_matrix_numpy(W, T):
    word_norms = np.linalg.norm(W, axis=1, keepdims=True)
    tag_norms = np.linalg.norm(T, axis=1, keepdims=True)

    if np.any(word_norms == 0) or np.any(tag_norms == 0):
        raise ValueError("Cosine similarity cannot be calculated for zero rows.")

    normal_words = W / word_norms
    normal_tags = T / tag_norms
    return normal_words @ normal_tags.T


def rank_tags(tag_names, S, in_vocab_tokens):
    S = np.asarray(S)

    if S.ndim != 2 or S.shape[1] != len(tag_names):
        raise ValueError("Similarity matrix dimensions do not match the tags.")
    if S.shape[0] != len(in_vocab_tokens):
        raise ValueError("Similarity matrix dimensions do not match the words.")
    if S.shape[0] == 0:
        raise ValueError("At least one in-vocabulary word is required.")

    best_word_indices = np.argmax(S, axis=0)
    highest_scores = np.max(S, axis=0)
    tag_order = np.argsort(-highest_scores)

    ranking = []
    for index in tag_order:
        best_word = in_vocab_tokens[best_word_indices[index]]
        ranking.append((tag_names[index], float(highest_scores[index]), best_word))

    return ranking


def main(text_path, model_path):
    model = load_model(model_path)

    with open(text_path, "r", encoding="utf-8") as text_file:
        raw_text = text_file.read()

    tokens_before = len(raw_text.split())
    processed_tokens = Vec.preprocess(raw_text)

    print("Tokens before preprocessing:", tokens_before)
    print("Tokens after preprocessing:", len(processed_tokens))

    tag_names, T = build_tag_matrix(model, TAGS)
    words, oov_words, W = build_text_matrix(model, processed_tokens)

    print("In-vocabulary tokens:", len(words))
    print("Out-of-vocabulary tokens:", len(oov_words))
    print("Distinct OOV tokens:", sorted(set(oov_words)))
    print("T shape:", T.shape)
    print("W shape:", W.shape)

    S_vector = similarity_matrix_vector_class(W, T)
    S_numpy = similarity_matrix_numpy(W, T)

    print("Similarity matrix shape:", S_numpy.shape)
    print("np.allclose:", np.allclose(S_vector, S_numpy, atol=1e-6))
    print("Maximum absolute difference:", np.max(np.abs(S_vector - S_numpy)))

    ranking = rank_tags(tag_names, S_numpy, words)
    print("\nTop eight tags")
    print(f"{'Rank':<6}{'Tag':<22}{'Score':<12}{'Best-matching word':<24}Match")
    print("-" * 76)

    for rank, (tag, score, word) in enumerate(ranking[:8], start=1):
        match = "identical" if tag == word else "different word"
        print(f"{rank:<6}{tag:<22}{score:<12.4f}{word:<24}{match}")


if __name__ == "__main__":
    text_path = "./text.txt"
    model_path = "../glove50/glove_50_fast.wordvectors"
    main(text_path, model_path)
