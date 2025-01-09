import logging

import numpy as np
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer

sts_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def get_ambiguous_words(words):
    logging.info(f'Checking ambiguous words from {words}')
    ambiguous_words = set()
    for w in words:
        synsets = wn.synsets(w, lang='por')
        if len(synsets) < 2:
            continue
        min_similarity = get_definitions_similarity(synsets)
        if min_similarity < 0.2:
            ambiguous_words.add((w, min_similarity))
    ambiguous_words = sorted(ambiguous_words, key=lambda x: x[1])
    return ambiguous_words

def get_definitions_similarity(synsets):
    logging.info(f'Calculating similarity between definitions of {synsets}')
    definitions = [s.definition() for s in synsets]
    embeddings = sts_model.encode(definitions)
    min_similarity = sts_model.similarity(embeddings, embeddings).min()
    return min_similarity

