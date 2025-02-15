from pathlib import Path

import polars as pl
from full_pun_generation.wordnet import sts_model
from transformers import pipeline

recognition_model = pipeline(
    "text-classification", "Superar/pun-recognition-pt")
results_path = Path("results/generation")
top_k_models = 2
top_k_jokes = 1
num_evaluators = 20
num_overlaps = 2


def semantic_similarity(group):
    embeddings = sts_model.encode([group["headline"], group["generated"]])
    similarity = sts_model.similarity(embeddings[0], embeddings[1])
    return similarity[0]


def typicality(puns):
    prediction = recognition_model(puns.to_list())
    return pl.Series([p["score"] for p in prediction])


def print_info(df, name):
    num_models = df["model"].n_unique()
    num_headlines = df["headline"].n_unique()
    num_inputs = df.n_unique(["headline", "pun sign", "alternative sign"])
    print(f"{name}: {df.height} puns from {num_models} models " +
          f"with {num_inputs} inputs from {num_headlines} headlines")


dfs = list()
for results_filepath in results_path.glob("*.jsonl"):
    print(f"Loading {results_filepath}")
    df = (pl.read_ndjson(results_filepath).unique()
          .with_columns(
              pl.lit(results_filepath.name).str.strip_suffix(".jsonl")
              .alias("model"))
          .select([pl.col("headline"),
                   pl.col("pun sign"),
                   pl.col("alternative sign"),
                   pl.col("model"),
                   pl.col("generated")
                   .str.extract(r"\{[^}]+\}", 0)
                   .str.extract(r"\"trocadilho\":\s?\"(.*)\"", 1)])
          )
    dfs.append(df)
print("**********")
df = pl.concat(dfs)
print_info(df, "First load")

# Remove examples with empty signs
null_signs = df.filter(pl.col('pun sign').is_null())
print_info(null_signs, "Null signs")
df = df.filter(pl.col('pun sign').is_not_null())
num_models = df["model"].n_unique()
print_info(df, "After removing null signs")

# If a model failed to generate pun for specific
# headline + pair of words, remove it for all models
failed = df.filter(pl.col("generated").is_null())
df = df.join(failed, how="anti",
             on=["headline", "pun sign", "alternative sign"])
print_info(failed, "Failed generations")
print_info(df, "After removing failed generations")
print("**********")

df = (df.filter(pl.col("generated").is_not_null())
        .with_columns(pl.struct([pl.col("headline"),
                                 pl.col("generated")])
                      .map_elements(semantic_similarity,
                                    return_dtype=pl.Float64)
                      .alias("similarity"))
        .with_columns(pl.col("generated")
                      .map_batches(typicality)
                      .alias("typicality"))
      )

# Get top models for each criterion
avg_df = (df.group_by("model")
          .agg([
              pl.col("similarity").mean().alias("avg similarity"),
              pl.col("typicality").mean().alias("avg typicality")
          ]))
top_similarity_models = (avg_df.sort("avg similarity", descending=True)
                         .head(top_k_models)["model"].to_list())
top_typicality_models = (avg_df.sort("avg typicality", descending=True)
                         .head(top_k_models)["model"].to_list())
worst_similarity_models = (avg_df.sort("avg similarity")
                           .head(top_k_models)["model"].to_list())
worst_typicality_models = (avg_df.sort("avg typicality")
                           .head(top_k_models)["model"].to_list())
top_models = set(top_similarity_models + top_typicality_models)
df = df.filter(pl.col("model").is_in(top_models))
print(f"Top similarity models: {top_similarity_models}")
print(f"Top typicality models: {top_typicality_models}")
print(f"Worst similarity models: {worst_similarity_models}")
print(f"Worst typicality models: {worst_typicality_models}")
print(f"Top models: {top_models}")
print_info(df, "After filtering top models")
print("**********")

# Get top-k puns for each criterion
top_similarity = (df.group_by(["model", "headline"])
                  .agg(pl.all().top_k_by(pl.col("similarity"),
                                         top_k_jokes)))
top_typicality = (df.group_by(["model", "headline"])
                  .agg(pl.all().top_k_by(pl.col("typicality"),
                                         top_k_jokes)))
df = (pl.concat([top_similarity, top_typicality])
      .explode(pl.all().exclude(["model", "headline"]))
      .unique()
      .with_row_index("id"))
print_info(df, "After selecting top puns")

# Create evaluation pairs
pairs = (df.join(df, how="cross")
         .filter(pl.col("model") != pl.col("model_right"))
         .filter(pl.col("headline") == pl.col("headline_right"))
         .filter(pl.col("id") < pl.col("id_right"))
         .sample(fraction=1.0, shuffle=True))
print(f"Number of contest pairs: {pairs.height}")

chunk_size = (pairs.height * num_overlaps) // num_evaluators
chunks = list(pairs.iter_slices(chunk_size))
print(f"Chunk sizes: {[chunk.height for chunk in chunks]}")

annotator_id = 0
for _ in range(num_overlaps):
    for chunk_id, chunk in enumerate(chunks):
        chunk = chunk.with_columns([pl.lit(annotator_id).alias("annotator_id"),
                                    pl.lit(chunk_id).alias("chunk_id")])
        chunk.write_ndjson(f"data/evaluation/annotator_{annotator_id}.jsonl")
        annotator_id += 1
