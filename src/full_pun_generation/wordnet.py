import logging

import numpy as np
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer

sts_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def get_ambiguous_words(words):
    logging.info(f"Checking ambiguous words from {words}")
    ambiguous_words = set()
    for w in words:
        logging.info(f"Checking word: {w}")
        synsets = wn.synsets(w, lang="por")
        if len(synsets) < 2:
            continue
        min_similarity, def1, def2 = get_definitions_similarity(synsets)
        if min_similarity < 0.2:
            ambiguous_words.add((w, min_similarity, def1, def2))
    ambiguous_words = sorted(ambiguous_words, key=lambda x: x[1])
    return ambiguous_words


def get_definitions_similarity(synsets1, synsets2=None):
    logging.info(f"Calculating similarity between definitions of {synsets1} and {synsets2}")

    definitions1 = [s.definition() for s in synsets1]
    definitions2 = [s.definition() for s in synsets2] if synsets2 else definitions1

    logging.info(f"Definitions #1: {definitions1}")
    if synsets2:
        logging.info(f"Definitions #2: {definitions2}")

    embeddings1 = sts_model.encode(definitions1)
    embeddings2 = sts_model.encode(definitions2) if synsets2 else embeddings1

    similarity = sts_model.similarity(embeddings1, embeddings2)
    min_index = np.unravel_index(similarity.argmin(), similarity.shape)
    min_similarity = similarity[min_index]

    definition1 = definitions1[min_index[0]]
    definition2 = definitions2[min_index[1]]

    logging.info(f"Most unsimilar: {min_similarity}, {definition1} - {definition2}")
    return min_similarity, definition1, definition2 


def get_valid_words(words):
    return [w for w in words if w in wn.words(lang="por")]


def get_words_synsets(words):
    return [wn.synsets(w, lang="por") for w in words]


def test():
    logging.basicConfig(level=logging.INFO)
    words = ["concelho", "zona", "vila", "português"]
    for w in words:
        synsets = wn.synsets(w, lang="por")
        logging.info(synsets)
        logging.info(get_definitions_similarity(synsets))
    ambiguous_words = get_ambiguous_words(words)
    logging.info(f"Ambiguous words: {ambiguous_words}")
