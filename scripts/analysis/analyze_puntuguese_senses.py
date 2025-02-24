import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from full_pun_generation.puntuguese import Puntuguese
from full_pun_generation.wordnet import (get_words_synsets,
                                         get_definitions_similarity)
import altair as alt


def get_similarities(words):
    synsets = get_words_synsets(words)
    min_similarities = list()
    for s in synsets:
        sim = np.inf
        if len(s) >= 2:
            sim, _, _ = get_definitions_similarity(s)
        min_similarities.append(sim)
    return pl.Series(min_similarities)


puntuguese = Puntuguese("../../Resources/Corpora/Puntuguese/data/puns.json")
puntuguese.filter_data()
homographs = (puntuguese.train.filter(pl.col("homograph"))
              .with_columns(
                  pl.col("pun sign")
                  .map_batches(get_similarities)
                  .alias("similarity"))
              .filter(~pl.col("similarity").is_infinite())
              )

print(homographs.select(pl.col("similarity"))
      .describe(percentiles=[0.25, 0.5, 0.7, 0.75]))

base = alt.Chart(homographs)
histogram = (base.mark_bar().encode(x=alt.X("similarity:Q").bin(),
                                    y="count()"))
rule = base.mark_rule(color="red").encode(x=alt.X(datum=0.2),
                                          size=alt.value(2))
chart = histogram + rule
chart.save("results/img/puntuguese_similarity_dist.png", ppi=300)

