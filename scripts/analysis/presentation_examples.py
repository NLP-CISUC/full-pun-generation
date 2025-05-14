from itertools import combinations

from full_pun_generation.context import expand_keywords, extract_keywords
from full_pun_generation.pronunciation import (get_pronunciation,
                                               phoneme_to_grapheme)
from full_pun_generation.wordnet import (get_ambiguous_words,
                                         get_definitions_similarity,
                                         get_valid_words, get_words_synsets)
from nltk.corpus import wordnet as wn

text = "O preço do Samsung Galaxy S25 cai a pique com os descontos na Samsung"

# Keyword extraction
print(f"Headline: {text}")
keywords = extract_keywords(text)
print(f"Extracted keywords: {keywords}")
expanded_keywords = expand_keywords(keywords)
print(f"Expanded keywords: {expanded_keywords}")

# Homographic words identification
words = {w for w, _ in expanded_keywords}
for w in words:
    synsets = wn.synsets(w, lang="por")
ambiguous_words = get_ambiguous_words(words)
print(f"Ambiguous words: {ambiguous_words}\n\n")


text = "A Semente do Figo Sagrado: o melhor filme do iraniano Mohammad Rasoulof"
keywords = extract_keywords(text)
expanded_keywords = expand_keywords(keywords)
words = {w for w, _ in expanded_keywords}
print(f"Headline: {text}")

# Homophonic words identification
homophones = []
phonemes = [pron for pron in get_pronunciation(words)]
print(f"Phonemes: {phonemes}")
graphemes = [phoneme_to_grapheme(pron)[1] for pron in phonemes]
print(f"Graphemes: {graphemes}")
graphemes = [get_valid_words(g) for g in graphemes]
graphemes = [g for g in graphemes if len(g) > 1]
print(f"Valid graphemes: {graphemes}")
for g in graphemes:
    for w1, w2 in combinations(g, 2):
        w1, w2 = str(w1), str(w2)
        if w1 == w2:
            continue
        synsets = get_words_synsets([w1, w2])
        _, def1, def2 = get_definitions_similarity(synsets[0], synsets[1])
        if [[w1, def1], [w2, def2]] in homophones:
            continue
        homophones.append([[w1, def1], [w2, def2]])
print(f"Homophones: {homophones}")
