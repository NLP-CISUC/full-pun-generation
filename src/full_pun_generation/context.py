import logging

from keybert import KeyBERT
from transformers import pipeline
from gensim.models import KeyedVectors

kw_model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')
pos_model = pipeline('ner', model='Emanuel/porttagger-base')
embeddings_model = KeyedVectors.load('../../Resources/Embeddings/Portuguese/glove_s300.kv')

def pos_tagging(text):
    logging.info('Performing POS tagging')
    doc = pos_model(text)
    pos_tags = [(str(ent['word']), str(ent['entity'])) for ent in doc]
    logging.info(f'POS tags: {pos_tags}')

    # Deal with subword tokens that start with '##'
    merged_tags = []
    i = 0
    while i < len(pos_tags):
        if not pos_tags[i][0].startswith('##'):
            current_word = pos_tags[i][0]
            current_tag = pos_tags[i][1]

            while i + 1 < len(pos_tags) and pos_tags[i + 1][0].startswith('##'):
                current_word += pos_tags[i + 1][0][2:]
                i += 1
            merged_tags.append((current_word, current_tag))
        i += 1
    logging.info(f'Merged POS tags: {merged_tags}')
    return merged_tags

def extract_keywords(text, n_keywords=5):
    logging.info(f'Extracting {n_keywords} keywords')
    logging.info(f'Input text: {text}')

    pos_tags = pos_tagging(text)
    stop_words = [word.lower() for word, tag in pos_tags
                  if tag not in ['NOUN', 'PROPN', 'ADJ', 'VERB', 'ADV']]
    logging.info(f'Stopwords: {stop_words}')

    keywords = kw_model.extract_keywords(text, top_n=n_keywords,
                                         stop_words=stop_words)
    logging.info(f'Keywords: {keywords}')
    return keywords

def expand_keywords(keywords):
    expanded_keywords = keywords.copy()
    for keyword, _ in keywords:
        if keyword not in embeddings_model:
            continue
        logging.info(f'Expanding keyword: {keyword}')
        similar_words = embeddings_model.most_similar(keyword, topn=5)
        expanded_keywords += similar_words
        logging.info(f'Similar words: {similar_words}')
    return expanded_keywords
