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
        min_similarity, _, _ = get_definitions_similarity(synsets)
        if min_similarity < 0.2:
            ambiguous_words.add((w, min_similarity))
    ambiguous_words = sorted(ambiguous_words, key=lambda x: x[1])
    return ambiguous_words


def get_definitions_similarity(synsets):
    logging.info(f"Calculating similarity between definitions of {synsets}")
    definitions = [s.definition() for s in synsets]
    logging.info(f"Definitions: {definitions}")
    embeddings = sts_model.encode(definitions)
    similarity = sts_model.similarity(embeddings, embeddings)
    min_index = np.unravel_index(similarity.argmin(), similarity.shape)
    min_similarity = similarity[min_index]
    logging.info(f"Most unsimilar: {min_similarity}, {min_index}")
    return min_similarity, definitions[min_index[0]], definitions[min_index[1]]


def get_valid_words(words):
    return [w for w in words if w in wn.words(lang="por")]


def test():
    logging.basicConfig(level=logging.INFO)
    words = ["concelho", "zona", "vila", "português"]
    for w in words:
        synsets = wn.synsets(w, lang="por")
        logging.info(synsets)
        logging.info(get_definitions_similarity(synsets))
    ambiguous_words = get_ambiguous_words(words)
    logging.info(f"Ambiguous words: {ambiguous_words}")
