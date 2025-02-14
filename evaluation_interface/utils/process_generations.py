from pathlib import Path

import polars as pl
import streamlit as st
from full_pun_generation.wordnet import sts_model
from transformers import pipeline


@st.cache_resource
def load_classifier():
    return pipeline("text-classification",
                    model="Superar/pun-recognition-pt")


def semantic_similarity(group):
    embeddings = sts_model.encode([group["headline"], group["generated"]])
    similarity = sts_model.similarity(embeddings[0], embeddings[1])
    return similarity[0]


def typicality(puns):
    prediction = load_classifier()(puns.to_list())
    return pl.Series([p["score"] for p in prediction])


@st.cache_data
def load_data():
    datapath = Path("../results/generation")
    dfs = list()
    for filepath in datapath.glob("*.jsonl"):
        model_name = filepath.stem
        df = (pl.read_ndjson(filepath)
              .select(pl.col("id").alias("headline_id"),
                      pl.col("headline"),
                      pl.col("pun sign"),
                      pl.col("alternative sign"),
                      pl.col("generated")
                      .str.extract(r"\{[^}]+\}", 0)
                      .str.extract(r"\"trocadilho\":\s?\"(.*)\"", 1),
                      pl.lit(model_name).alias("model")))
        dfs.append(df)
    df = pl.concat(dfs)

    # Remove examples with empty signs
    df = df.filter(pl.col('pun sign').is_not_null())

    # Remove examples with failed generation
    df = df.filter(pl.col("generated").is_not_null())

    # Calculate metrics
    df = (df.filter(pl.col("generated").is_not_null())
            .with_columns(pl.struct([pl.col("headline"),
                                     pl.col("generated")])
                          .map_elements(semantic_similarity,
                                        return_dtype=pl.Float64)
                          .alias("similarity"))
            .with_columns(pl.col("generated")
                          .map_batches(typicality)
                          .alias("typicality"))
            .with_columns((0.5 * pl.col("similarity") + 0.5 * pl.col("typicality"))
                          .alias("score"))
            .sort("score", descending=True)
            .group_by(pl.col("headline_id", "model"))
            .first()
          )
    return df
