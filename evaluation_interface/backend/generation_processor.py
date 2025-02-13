import random
from pathlib import Path

import polars as pl
from full_pun_generation.wordnet import sts_model
from transformers import generation, pipeline

# Global dictionary to store pre-processed evaluation data.
evaluation_dataset = None

classifier = pipeline("text-classification",
                      model="Superar/pun-recognition-pt")


def semantic_similarity(group):
    embeddings = sts_model.encode([group["headline"], group["generated"]])
    similarity = sts_model.similarity(embeddings[0], embeddings[1])
    return similarity[0]


def typicality(puns):
    prediction = classifier(puns.to_list())
    return pl.Series([p["score"] for p in prediction])


def process_generation_files():
    generation_dir = Path("../results/generation")
    dfs = list()
    for file_path in generation_dir.glob("*.jsonl"):
        model_name = file_path.stem
        df = (pl.read_ndjson(file_path)
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
          )

    global evaluation_dataset
    evaluation_dataset = (df.sort("score", descending=True)
                          .group_by(["headline_id", "model"])
                          .first())


def get_all_headlines():
    if evaluation_dataset is None:
        process_generation_files()
    return (evaluation_dataset
            .select(pl.col("headline_id"),
                    pl.col("headline"))
            .unique()
            .to_dicts())


def get_generated_for_headline(headline_id):
    if evaluation_dataset is None:
        process_generation_files()
    return (evaluation_dataset.filter(pl.col("headline_id") == int(headline_id))
            .to_dicts())


def get_evaluation_data():
    if evaluation_dataset is None:
        process_generation_files()
    return evaluation_dataset.iter_rows(named=True)
