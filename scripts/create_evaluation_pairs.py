from pathlib import Path

import polars as pl
from full_pun_generation.wordnet import sts_model
from transformers import pipeline

recognition_model = pipeline(
    "text-classification", "Superar/pun-recognition-pt")
results_path = Path("results/generation")
top_k = 2
num_evaluators = 20
num_overlaps = 3


def semantic_similarity(group):
    embeddings = sts_model.encode([group["headline"], group["generated"]])
    similarity = sts_model.similarity(embeddings[0], embeddings[1])
    return similarity[0]


def typicality(puns):
    prediction = recognition_model(puns.to_list())
    return pl.Series([p["score"] for p in prediction])


dfs = list()
for results_filepath in results_path.glob("*.jsonl"):
    df = (pl.read_ndjson(results_filepath)
          .with_columns(
              pl.lit(results_filepath.name).str.strip_suffix(".jsonl")
              .alias("model"))
          .select([pl.col("headline"),
                   pl.col("pun sign"),
                   pl.col("alternative sign"),
                   pl.col("model"),
                   pl.col("generated")
                   .str.extract(r"\{[^}]+\}", 0)
                   .str.extract(r"\"trocadilho\": \"(.*)\"", 1)])
          )
    dfs.append(df)

df = (pl.concat(dfs)
        .filter(pl.col("generated").is_not_null())
        .with_columns(pl.struct([pl.col("headline"),
                                 pl.col("generated")])
                      .map_elements(semantic_similarity,
                                    return_dtype=pl.Float64)
                      .alias("similarity"))
        .with_columns(pl.col("generated")
                      .map_batches(typicality)
                      .alias("typicality"))
      )

# Get top-k puns for each criterion
top_similarity = (df.group_by("model")
                  .agg(pl.all().top_k_by(pl.col("similarity"), top_k)))
top_typicality = (df.group_by("model")
                  .agg(pl.all().top_k_by(pl.col("typicality"), top_k)))

df = (pl.concat([top_similarity, top_typicality])
      .explode(pl.all().exclude("model"))
      .unique()
      .with_row_index("id"))

# Create evaluation pairs
pairs = (df.join(df, how="cross")
         .filter(pl.col("model") != pl.col("model_right"))
         .filter(pl.col("id") < pl.col("id_right"))
         .sample(fraction=1.0, shuffle=True))

chunk_size = pairs.height // num_evaluators
chunks = pairs.iter_slices(chunk_size)

annotator_id = 0
for _ in range(num_overlaps):
    for chunk_id, chunk in enumerate(chunks):
        chunk = chunk.with_columns([pl.lit(annotator_id).alias("annotator_id"),
                                    pl.lit(chunk_id).alias("chunk_id")])
        chunk.write_ndjson(f"data/evaluation/annotator_{annotator_id}.jsonl")
        annotator_id += 1

