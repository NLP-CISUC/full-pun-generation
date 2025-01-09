import logging
import re
from itertools import product

from nltk.corpus import wordnet as wn
from phonemizer import phonemize
from phonemizer.separator import Separator
from tqdm import trange

# Phoneme to grapheme mapping for Portuguese
p2g = {'a': {'a', 'á', 'à', 'ha', 'há'}, 'ã': {'ã', 'am', 'an', 'hã', 'ham', 'han'},
       'aː': {'à'}, 'ɐ̃': {'ã', 'am', 'an', 'hã', 'ham', 'han', 'a', 'â', 'ha'},
       'æ': {'a'}, 'ɑ': {'a'}, 'o': {'o', 'ô', 'ho'}, 'õ': {'õ', 'om', 'on'},
       'ɔ': {'o', 'ó', 'ho', 'hó'}, 'ɔ̃': {'õ', 'om', 'on'}, 'u': {'u', 'ú', 'hu', 'hú'},
       'ũ': {'u', 'um', 'un', 'ú', 'hu', 'hú'},'ʊ': {'o', 'u'}, 'aɪ': {'ai', 'hai'},
       'aʊ': {'au', 'al', 'áu', 'ao', 'ál'}, 'ɐ̃ʊ̃': {'ão', 'am'}, 'oɪ': {'oi'},
       'oʊ': {'ou'},'ɔɪ': {'ói', 'oi'}, 'uɪ': {'ui', 'ue'},'w': {'u', 'w','l'},
       'e': {'e', 'ê', 'he'}, 'ɛ': {'e', 'é','a'}, 'i': {'i', 'í', 'y', 'e', 'hi'},
       'ĩ': {'i', 'im', 'in', 'ím', 'ín'}, 'iː': {'i', 'í', 'y', 'e'},
       'ɪ': {'i', 'y'},  'eɪ': {'e', 'é', 'ei', 'ê'}, 'eʊ': {'eu', 'el', 'eo'},
       'ɛɪ': {'éi', 'ei'}, 'ɛʊ': {'éu', 'el', 'eo', 'hel'}, 'iʊ': {'io', 'iu', 'il'},
       'y': {'e', 'i', 'y'}, 'j': {'i', 'e', 'h'}, 'ə': {''}, 'p': {'p'},
       'b': {'b'}, 't': {'t'}, 'd': {'d'}, 'f': {'f', 'ph'}, 'v': {'v'}, 'ɾ': {'r'},
       'r': {'r'}, 'ɹ': {'r'}, 'm': {'m'}, 'n': {'n'}, 'l': {'l', 'lh'}, 'dʒ': {'d'},
       'tʃ': {'t', 'tch'}, 'k': {'c', 'k', 'qu', 'q', 'ck'},
       'c': {'c', 'k'}, 'x': {'r', 'rr', 'h'}, 'h': {'r', 'rr', 'h'},
       'ɡ': {'g', 'gu'}, 'ŋ': {'m', 'n', '', 'ng'}, 'z': {'s', 'z', 'x'},
       's': {'s', 'ss', 'ç', 'sç', 'ss', 'c', 'sc', 'x', 'xc', 'z'},
       'ʃ': {'ch', 'x', 's', 'z', 'sh'}, 'ʒ': {'j', 'g'}, 'ɲ': {'nh'},
       'ʎ': {'lh', 'li'}, 'ts': {'tc', 'ts', 'zz'}, 'ks': {'x', 'cç', 'cs', 'cc'}}
phonetic_vowels = set(list(p2g.keys())[:35])
phonetic_aou_vowels = set(list(p2g.keys())[:21])
phonetic_ei_vowels = phonetic_vowels - phonetic_aou_vowels
phonetic_consonants = set(list(p2g.keys())[35:])

graphic_vowels = {'a', 'á', 'à', 'ã', 'â', 'e', 'é', 'ê', 'i', 'í', 'y', 'o',
                  'ó', 'ô', 'õ', 'u', 'ú', 'ú'}

def generate_all_possibilities(graphemes, preffix=''):
    if len(graphemes) == 0:
        return [preffix]

    # Orthographic rules
    # Only one accented vowel per word
    accent_vowels = {'á', 'â', 'é', 'ê', 'í', 'ó', 'ô', 'ú'}
    if re.search(rf'[{"".join(accent_vowels)}]', preffix):
        graphemes[0] = graphemes[0] - accent_vowels
    # No 'h' after 'h'
    if preffix.endswith('h'):
        start_with_h = {graph for graph in graphemes[0] if graph.startswith('h')}
        graphemes[0] = graphemes[0] - start_with_h
    # No repeating consonants
    if preffix and preffix[-1] not in graphic_vowels and preffix[-1] == graphemes[0]:
        start_with_last = {graph for graph in graphemes[0] if graph.startswith(preffix[-1])}
        graphemes[0] = graphemes[0] - start_with_last

    new_preffixes = [''.join(p) for p in product([preffix], graphemes[0])]
    possibilities = list()
    for p in new_preffixes:
        possibilities += generate_all_possibilities(graphemes[1:], preffix=p)
    print(possibilities)
    return possibilities

def phoneme_to_grapheme(pronunciation):
    logging.info(f'Generating graphemes for: {pronunciation}')
    phonemes = pronunciation.replace(' ', '|')
    phonemes = phonemes.replace('k|s', 'ks')
    phonemes = phonemes.replace('l|j', 'ʎ')
    phonemes = phonemes.split('|')
    graphemes = [p2g[p] if p in p2g else {'-'} for p in phonemes]

    # Phonotactic rules
    for i in range(len(graphemes)):
        start_with_h = {graph for graph in graphemes[i] if graph.startswith('h')}

        if i == 0:
            if phonemes[i] in {'h', 'x'}:
                graphemes[i] = graphemes[i] - {'rr'}
            if phonemes[i] == 's':
                graphemes[i] = graphemes[i] - {'ss', 'sç', 'ç'}
            if phonemes[i] == 'k':
                graphemes[i] = graphemes[i] - {'ck'}
        if i > 0:
            if phonemes[i] == 'ŋ' and phonemes[i-1] not in {'i'}:
                graphemes[i] = graphemes[i] - {''}
            if start_with_h:
                graphemes[i] = graphemes[i] - start_with_h
        if i > 0 and i < len(graphemes)-1:
            if phonemes[i] in {'h', 'x'} and phonemes[i-1] in phonetic_vowels and phonemes[i+1] in phonetic_vowels:
                graphemes[i] = graphemes[i] - {'r'}
        if i < len(graphemes)-1:
            if phonemes[i] == 'tʃ' and phonemes[i+1] in phonetic_aou_vowels: graphemes[i] = graphemes[i] - {'t'}
            if phonemes[i] == 'dʒ' and phonemes[i+1] in phonetic_aou_vowels:
                graphemes[i] = graphemes[i] - {'d'}
            if phonemes[i] == 'k' and phonemes[i+1] in phonetic_aou_vowels:
                graphemes[i] = graphemes[i] - {'qu'}
            if phonemes[i] == 'k' and phonemes[i+1] in phonetic_ei_vowels:
                graphemes[i] = graphemes[i] - {'c'}
            if phonemes[i] == 'k' and phonemes[i+1] not in ['w']:
                graphemes[i] = graphemes[i] - {'q'}
            if phonemes[i] == 's' and phonemes[i+1] not in phonetic_ei_vowels:
                graphemes[i] = graphemes[i] - {'sc', 'c', 'xc'}
            if phonemes[i] == 's' and phonemes[i+1] not in phonetic_aou_vowels:
                graphemes[i] = graphemes[i] - {'ç', 'sç'}
            if phonemes[i] == 's' and phonemes[i+1] in phonetic_vowels:
                graphemes[i] = graphemes[i] - {'z'}
            if phonemes[i] == 's' and phonemes[i+1] in phonetic_consonants:
                graphemes[i] = graphemes[i] - {'ss'}
            if phonemes[i] == 'ŋ' and graphemes[i+1].intersection({'p', 'b'}):
                graphemes[i] = graphemes[i] - {'n'}
            if phonemes[i] == 'ŋ' and not graphemes[i+1].intersection({'p', 'b'}):
                graphemes[i] = graphemes[i] - {'m'}
            if phonemes[i] == 'eɪ' and not phonemes[i+1] in {'ŋ', 'm', 'n'}:
                graphemes[i] = {'ei'}
            if phonemes[i] == 'ks' and phonemes[i+1] not in phonetic_aou_vowels:
                graphemes[i] = graphemes[i] - {'cç'}
            if phonemes[i] == 'ks' and phonemes[i+1] not in phonetic_ei_vowels:
                graphemes[i] = graphemes[i] - {'cc'}
            if phonemes[i] in {'ã', 'ɐ̃'} and phonemes[i+1] in {'ŋ', 'm', 'n'}:
                graphemes[i] = graphemes[i] - {'am', 'an'}
            if phonemes[i] == 'ŋ':
                graphemes[i] = graphemes[i] - {'ng'}
            if phonemes[i] == 'ʃ':
                graphemes[i] = graphemes[i] - {'s', 'z'}
        if i == len(graphemes)-1:
            if phonemes[i] == 's':
                graphemes[i] = graphemes[i] - {'ç', 'sç', 'c', 'sc', 'x', 'xc'}
            if phonemes[i] == 'k':
                graphemes[i] = graphemes[i] - {'qu', 'q'}
        if 'à' in graphemes[i] and len(phonemes) > 1:
            graphemes[i] = graphemes[i] - {'à'}
    all_writings = generate_all_possibilities(graphemes)

    # Keep only valid words
    all_writings = [w for w in all_writings if w in wn.words(lang='por')]
    if not all_writings:
        return []
    all_prons = get_pronunciation(all_writings)
    valid_writings = [w for w, w_pron in zip(all_writings, all_prons)
                      if w_pron == pronunciation]
    return valid_writings

def get_pronunciation(words):
    logging.info(f'Getting pronunciation for: {words}')
    phn = phonemize(words, language='pt-br', backend='espeak', strip=True,
                    separator=Separator(phone='|', word=' ', syllable='.'),
                    njobs=4)
    return phn

if __name__ == '__main__':
    from nltk.corpus import floresta
    corpus = list(set(floresta.words()))

    start = 0
    for i in trange(start, len(corpus), initial=start):
        word = corpus[i].lower()
        if word == 'é':
            continue
        if '_' in word:
            continue
        if '.' in word:
            continue
        if '-' in word:
            continue
        if ',' in word:
            continue
        if "'" in word:
            continue
        if re.search(r'\d', word):
            continue
        if re.search(r'kh', word):
            continue
        if re.search(r'pp', word):
            continue
        if word in ['petting', 'rmg', 'jn', 'dn', 'rv', 'chahine', 'pozzi',
                    'tahar', 'urss', 'tmg', 'urrah', 'pcp', 'ps', 'br',
                    'collabra', 'software', 'oms', 'libreville', 'bd', 'gotti',
                    'sites', 'terràvista', 'site', 'newsweek', 'cgtp', 'vw',
                    'qca', 'ft', 'figgis', 'délèze', 'o{s}', 'pf', 'benetton',
                    'seabrook']:
            continue
        pronunciation = get_pronunciation([word])[0]

        if not pronunciation:
            continue
        all_writings = phoneme_to_grapheme(pronunciation)

        if '-' in word:
            word = word.replace('-', '')
        if word in all_writings:
            continue

        print(f'{word} -> {pronunciation}')
        print(all_writings)
        break
