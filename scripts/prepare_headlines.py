import polars as pl
from full_pun_generation.context import expand_keywords, extract_keywords
from full_pun_generation.pronunciation import (get_pronunciation,
                                               phoneme_to_grapheme)
from full_pun_generation.wordnet import get_ambiguous_words, get_valid_words


def get_signs(text, n_keywords=5):
    keywords = extract_keywords(text, n_keywords)
    expanded_keywords = expand_keywords(keywords)
    words = {kw for kw, _ in expanded_keywords}

    homographic_signs = get_ambiguous_words(words)
    graphemes = [phoneme_to_grapheme(pron)[1]
                 for pron in get_pronunciation(words)]
    graphemes = [get_valid_words(g) for g in graphemes]
    homophonic_signs = [g for g in graphemes if len(g) > 1]
    return {"homographic": "\n".join([str(w) for w, _ in homographic_signs]),
            "homophonic": "\n".join([str(g) for g in homophonic_signs])}


df = (pl.read_ndjson("data/headlines.jsonl").head(1)
      .with_columns(
          pl.col("headline").map_elements(
              get_signs,
              return_dtype=pl.Struct({"homographic": pl.String,
                                      "homophonic": pl.String})
          )))
print(df)
