import logging
from pathlib import Path
import re
from itertools import product

from nltk.corpus import wordnet as wn
from phonemizer import phonemize
from phonemizer.separator import Separator
from tqdm import trange

# Phoneme to grapheme mapping for Portuguese
p2g = {'a': {'a', 'á', 'à', 'ha', 'há'}, 'ã': {'ã', 'am', 'an', 'hã', 'ham', 'han'},
       'aː': {'à'}, 'ɐ̃': {'ã', 'am', 'an', 'hã', 'ham', 'han', 'a', 'â', 'ha'},
       'æ': {'a', 'â'}, 'ɑ': {'a'}, 'o': {'o', 'ô', 'ho', 'oo'}, 'õ': {'õ', 'om', 'on'},
       'ɔ': {'o', 'ó', 'ho', 'hó'}, 'ɔ̃': {'õ', 'om', 'on'}, 'u': {'u', 'ú', 'hu', 'hú'},
       'ũ': {'u', 'um', 'un', 'ú', 'hu', 'hú'},'ʊ': {'o', 'u', 'l', 'oo'},
       'aɪ': {'ai', 'ay', 'hai', 'hay'}, 'aʊ': {'au', 'al', 'áu', 'ao', 'ál'},
       'ɐ̃ʊ̃': {'ão', 'am'}, 'oɪ': {'oi'}, 'oʊ': {'ou'},'ɔɪ': {'ói', 'oi', 'oy'},
       'uɪ': {'ui', 'ue'},'w': {'u', 'w', 'l'}, 'e': {'e', 'ê', 'he'},
       'ɛ': {'e', 'é','a', 'hé'}, 'i': {'i', 'í', 'y', 'e', 'hi', 'hí', 'hy'},
       'ĩ': {'i', 'im', 'in', 'ím', 'ín'}, 'iː': {'i', 'í', 'y', 'e'},
       'ɪ': {'i', 'í', 'y'},  'eɪ': {'e', 'é', 'ei', 'ê', 'ey', 'ay', 'a'},
       'eʊ': {'eu', 'el', 'eo', 'êu'}, 'ɛɪ': {'éi', 'ei'},
       'ɛʊ': {'éu', 'el', 'él', 'eo', 'hel', 'hél'}, 'iʊ': {'io', 'iu', 'il', 'hil'},
       'y': {'e', 'i', 'y'}, 'j': {'i', 'e', 'h', 'y'}, 'ə': {''}, 'p': {'p'},
       'b': {'b'}, 't': {'t', 'th'}, 'd': {'d'}, 'f': {'f', 'ph'}, 'v': {'v', 'w'}, 'ɾ': {'r'},
       'r': {'r'}, 'ɹ': {'r'}, 'm': {'m'}, 'n': {'n'}, 'l': {'l', 'lh', 'll'}, 'dʒ': {'d'},
       'tʃ': {'t', 'tch'}, 'k': {'c', 'k', 'qu', 'q', 'ck', 'ch'},
       'c': {'c', 'k'}, 'x': {'r', 'rr', 'h'}, 'h': {'r', 'rr', 'h'},
       'ɡ': {'g', 'gu'}, 'ŋ': {'m', 'n', '', 'ng'}, 'z': {'s', 'z', 'x'},
       's': {'s', 'ss', 'ç', 'sç', 'ss', 'c', 'sc', 'x', 'xc', 'z'},
       'ʃ': {'ch', 'x', 's', 'z', 'sh'}, 'ʒ': {'j', 'g'}, 'ɲ': {'nh'},
       'ʎ': {'lh', 'li'}, 'ts': {'tc', 'ts', 'zz'}, 'ks': {'x', 'cç', 'cs', 'cc', 'ks', 'kss'}}
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
    # No 'ql' for 'qu'
    if preffix.endswith('q'):
        start_with_l = {graph for graph in graphemes[0] if graph.startswith('l')}
        graphemes[0] = graphemes[0] - start_with_l

    new_preffixes = [''.join(p) for p in product([preffix], graphemes[0])]
    possibilities = list()
    for p in new_preffixes:
        possibilities += generate_all_possibilities(graphemes[1:], preffix=p)
    return possibilities

def phoneme_to_grapheme(pronunciation):
    logging.info(f'Generating graphemes for: {pronunciation}')
    phonemes = pronunciation.replace('ˌ', '')
    phonemes = phonemes.replace(' ', '|')
    phonemes = phonemes.replace('k|s', 'ks')
    phonemes = phonemes.replace('l|j', 'ʎ')
    phonemes = phonemes.replace('t|ʃ', 'tʃ')
    phonemes = phonemes.split('|')

    stress = [True if 'ˈ' in p else False for p in phonemes]
    phonemes = [p.replace('ˈ', '') for p in phonemes]
    graphemes = [p2g[p] if p in p2g else {'-'} for p in phonemes]

    # Mapping rules
    for i in range(len(graphemes)):
        start_with_h = {graph for graph in graphemes[i] if graph.startswith('h')}

        if i == 0:
            if phonemes[i] in {'h', 'x'}:
                graphemes[i] = graphemes[i] - {'rr'}
            if phonemes[i] == 's':
                graphemes[i] = graphemes[i] - {'ss', 'sç', 'ç', 'x', 'xc'}
            if phonemes[i] == 'k':
                graphemes[i] = graphemes[i] - {'ck'}
            if phonemes[i] == 'w' or phonemes[i] == 'ʊ':
                graphemes[i] = graphemes[i] - {'l'}
            if phonemes[i] == 'l':
                graphemes[i] = graphemes[i] - {'lh'}
        if i > 0:
            if phonemes[i] == 'ŋ' and phonemes[i-1] not in {'i', 'ɐ̃'}:
                graphemes[i] = graphemes[i] - {''}
            if start_with_h:
                graphemes[i] = graphemes[i] - start_with_h
        if i > 0 and i < len(graphemes)-1:
            if phonemes[i] in {'h', 'x'} and phonemes[i-1] in phonetic_vowels and phonemes[i+1] in phonetic_vowels:
                graphemes[i] = graphemes[i] - {'r'}
        if i < len(graphemes)-1:
            if phonemes[i] == 'tʃ' and phonemes[i+1] in phonetic_aou_vowels:
                graphemes[i] = graphemes[i] - {'t'}
            if phonemes[i] == 'dʒ' and phonemes[i+1] in phonetic_aou_vowels:
                graphemes[i] = graphemes[i] - {'d'}
            if phonemes[i] == 'ʒ' and phonemes[i+1] in phonetic_aou_vowels:
                graphemes[i] = graphemes[i] - {'g'}
            if phonemes[i] == 'ɡ' and phonemes[i+1] in phonetic_consonants:
                graphemes[i] = graphemes[i] - {'gu'}
            if phonemes[i] == 'k' and phonemes[i+1] in phonetic_aou_vowels:
                graphemes[i] = graphemes[i] - {'qu'}
            if phonemes[i] == 'k' and phonemes[i+1] in phonetic_ei_vowels:
                graphemes[i] = graphemes[i] - {'c'}
            if phonemes[i] == 'k' and phonemes[i+1] not in ['w']:
                graphemes[i] = graphemes[i] - {'q'}
            if phonemes[i] == 'k' and phonemes[i+1] == 'r':
                graphemes[i] = graphemes[i] - {'q', 'qu'}
            if phonemes[i] == 'k' and phonemes[i+1] == 'ɾ':
                graphemes[i] = graphemes[i] - {'q', 'qu'}
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
                graphemes[i] = {'ei', 'ay', 'a'}
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
                graphemes[i] = graphemes[i] - {'ç', 'sç', 'c', 'sc', 'xc'}
            if phonemes[i] == 'k':
                graphemes[i] = graphemes[i] - {'qu', 'q'}
            if phonemes[i] == 'ks':
                graphemes[i] = graphemes[i] - {'cç', 'cc'}
        if 'à' in graphemes[i] and len(phonemes) > 1:
            graphemes[i] = graphemes[i] - {'à'}
        if any(stress) and not stress[i]:
            graphemes[i] = graphemes[i] - {'á', 'é', 'í', 'ó', 'ú', 'â', 'ê', 'ô',
                                           'êu', 'éi', 'ói', 'hú', 'áu', 'ál', 'hí',
                                           'ím', 'ín', 'éu', 'él', 'hé', 'hél'}
    all_writings = generate_all_possibilities(graphemes)

    # Keep only recreations that are pronounced the same
    all_prons = get_pronunciation(all_writings)
    valid_writings = [w for w, w_pron in zip(all_writings, all_prons)
                      if w_pron == pronunciation]
    return all_writings, valid_writings

def get_pronunciation(words):
    logging.info(f'Getting pronunciation for: {words}')
    phn = phonemize(words, language='pt-br', backend='espeak', strip=True,
                    separator=Separator(phone='|', word=' ', syllable='.'),
                    with_stress=True, njobs=4)
    return phn

def main():
    from nltk.corpus import floresta
    corpus = {word.strip().lower() for word in floresta.words()}

    ignore_filepath = Path('data/ignore_pron_words.txt')
    ignore_file = ignore_filepath.open('r+')
    ignore_words = {line.strip() for line in ignore_file if line.strip()
                    and not line.startswith('#')}
    corpus = list(corpus - ignore_words)

    for i in trange(0, len(corpus), initial=0):
        word = corpus[i]
        pronunciation = get_pronunciation([word])[0]

        if not pronunciation:
            print(f'No pronunciation found for {word}')
            continue
        all_writings, _ = phoneme_to_grapheme(pronunciation)

        if '-' in word:
            word = word.replace('-', '')
        if word in all_writings:
            ignore_file.write(f'{word}\n')
            continue

        print(f'{word} -> {pronunciation}')
        print(all_writings)
        print('----------------------------------------')
    ignore_file.close()

if __name__ == '__main__':
    main()
