from itertools import combinations

import polars as pl
from full_pun_generation.context import expand_keywords, extract_keywords
from full_pun_generation.pronunciation import (get_pronunciation,
                                               phoneme_to_grapheme)
from full_pun_generation.wordnet import (get_ambiguous_words,
                                         get_valid_words,
                                         get_words_synsets,
                                         get_definitions_similarity)


def get_signs(text, n_keywords=5):
    keywords = extract_keywords(text, n_keywords)
    expanded_keywords = expand_keywords(keywords)
    words = {kw for kw, _ in expanded_keywords}

    homographic_signs = get_ambiguous_words(words)
    signs = [[[str(w), str(def1)], [str(w), str(def2)]]
             for w, _, def1, def2 in homographic_signs if w]

    graphemes = [phoneme_to_grapheme(pron)[1]
                 for pron in get_pronunciation(words)]
    graphemes = [get_valid_words(g) for g in graphemes]
    graphemes = [g for g in graphemes if len(g) > 1]
    for g in graphemes:
        for w1, w2 in combinations(g, 2):
            w1, w2 = str(w1), str(w2)
            if w1 == w2:
                continue
            synsets = get_words_synsets([w1, w2])
            _, def1, def2 = get_definitions_similarity(synsets[0], synsets[1])
            if [[w1, def1], [w2, def2]] in signs:
                continue
            signs.append([[w1, def1], [w2, def2]])
    return signs


df = (pl.read_ndjson("data/headlines.jsonl")
      .with_columns(
          pl.col("headline").map_elements(
              get_signs,
              return_dtype=pl.List(pl.List(pl.List(pl.String)))
          ).alias("signs"))
      .explode("signs")
      .with_columns(
          pl.col("signs").list.get(0).list.get(0).alias("pun sign"),
          pl.col("signs").list.get(1).list.get(0).alias("alternative sign"),
          pl.col("signs").list.get(0).list.get(1).alias("pun definition"),
          pl.col("signs").list.get(1).list.get(1).alias("alternative definition"))
      .drop("signs")
      )
df.write_ndjson("data/processed_headlines.jsonl")
