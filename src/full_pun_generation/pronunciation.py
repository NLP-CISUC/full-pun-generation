import logging
from itertools import product

from phonemizer import phonemize
from phonemizer.separator import Separator

p2g = {'a': ['a', 'á', 'à'],
       'ɐ̃': ['a', 'ã', 'â', 'an', 'am', 'ân', 'âm'],
       'æ': ['a', 'â'],
       'e': ['e', 'ê'],
       'ɛ̃': ['e', 'ẽ', 'en', 'em', 'ên', 'êm'],
       'ɛ': ['e', 'é'],
       'i': ['i', 'í'],
       'ɪ': ['e', 'i', 'y'],
       'y': ['e', 'i', 'y'],
       'o': ['o', 'ô'],
       'ɔ': ['o', 'ó'],
       'u': ['u', 'ú'],
       'ʊ': ['o', 'u'],
       'w': ['u', 'w', 'l'],
       'ə': [''],
       'aɪ': ['ai'],
       'aʊ': ['au'],
       'eɪ': ['e', 'ei'],
       'b': ['b'],
       'd': ['d'],
       'dʒ': ['d'],
       'f': ['f'],
       'ʒ': ['g', 'j', 's', 'x', 'z'],
       'l': ['l'],
       'm': ['m'],
       'n': ['n'],
       'ŋ': ['n', 'm'],
       'p': ['p'],
       'r': ['r'],
       'ɾ': ['r'],
       's': ['ç', 's', 'ss', 'x', 'z'],
       't': ['t'],
       'tʃ': ['t']
       }

def get_pronunciation(words):
    phn = phonemize(words, language='pt-br', backend='espeak', strip=True,
                    separator=Separator(phone='|', word=' ', syllable='.'),
                    njobs=4)
    logging.info(f'Pronunciation: {phn}')
    return phn

if __name__ == '__main__':
    pronounces = get_pronunciation(['todos', 'os', 'seres', 'humanos', 'nascem', 'livres', 'e', 'iguais', 'em', 'dignidade', 'e', 'direitos', 'são', 'dotados', 'de', 'razão', 'e', 'consciência', 'e', 'devem', 'agir', 'uns', 'para', 'com', 'os', 'outros', 'em', 'espírito', 'de', 'fraternidade'])
    for word in pronounces:
        all_writings = [p2g[p] if p in p2g else ['-'] for p in word.split('|')]
        all_writings = [''.join(w) for w in product(*all_writings)]

        must_print = any(['-' in w for w in all_writings])
        if must_print:
            print(f'{word} -> {all_writings}')
