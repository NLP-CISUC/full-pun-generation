from argparse import ArgumentParser
from pathlib import Path

from polars.convert import normalize

import altair as alt
import krippendorff
import numpy as np
import polars as pl
import yaml

parser = ArgumentParser()
parser.add_argument("--charts", action="store_true")
args = parser.parse_args()

fun_order = ["Não tem piada", "Tem pouca piada", "Tem piada"]
rel_order = ["Não tem relação", "Tem pouca relação", "Tem relação"]
model_names = {"deepseek-r1-70b": "Deepseek",
               "llama3-3": "Llama3.3",
               "ptt5-v2": "PTT5",
               "_definitions": "+def",
               "_fewshot": "+shot"}

bar_color_scheme = ["#ffa19b", "#fedd90", "#6ab7c7"]
text_color_scheme = ["#813232", "#6c570c", "#004f5d"]


results_path = Path("results/evaluation")
dfs = [pl.read_ndjson(f) for f in results_path.glob("*.jsonl")]
df = (pl.concat(dfs)
      .select([pl.col("evaluator"),
               pl.col("model").str.replace_many(model_names),
               pl.col("headline_id"),
               pl.col("headline"),
               pl.col("generated"),
               pl.col("funniness"),
               pl.col("typicality"),
               pl.col("relation"),
               pl.col("similarity")]))

# Due to the way the data was collected, if an evaluator only
# passed through the headline, not changing the evaluation from
# the default, it was not recorded. This code adds those rows.
new_rows = list()
splits_path = Path("evaluation_interface/config/config.yaml")
splits = yaml.safe_load(splits_path.read_text())["splits"]
splits_df = (pl.DataFrame([{"evaluator": evaluator, "headline_id": headline_ids}
                           for evaluator, headline_ids in splits.items()
                           if evaluator in df["evaluator"].unique()])
             .explode("headline_id"))
splits_df = splits_df.join(df.select(pl.col("model")), how="cross").unique()
missing_df = (splits_df.join(df, on=["evaluator", "headline_id", "model"], how="anti")
              .with_columns([pl.lit("Não tem piada").alias("funniness"),
                             pl.lit("Não tem relação").alias("relation")])
              .join(df.select(pl.all().exclude(["evaluator", "funniness", "relation"])).unique(),
                    on=["headline_id", "model"], how="left")
              .select(pl.col("evaluator"),
                      pl.col("model"),
                      pl.col("headline_id"),
                      pl.col("headline"),
                      pl.col("generated"),
                      pl.col("funniness"),
                      pl.col("typicality"),
                      pl.col("relation"),
                      pl.col("similarity")))
print(f"Added {missing_df.height} missing rows.\n")
df = (df.extend(missing_df)
      .with_columns(pl.col("funniness").cast(pl.Enum(fun_order)).to_physical() + 1,
                    pl.col("relation").cast(pl.Enum(rel_order)).to_physical() + 1))

# Descriptive statistics
agg_df = (df.group_by(["model", "headline_id"])
          .agg(pl.col("funniness").mode().max(),
               pl.col("typicality").first(),
               pl.col("relation").mode().max(),
               pl.col("similarity").first()))


print("###### Descriptive statistics ######")

def tastle_wierman_consensus(x: pl.Series) -> float:
    x_idx = x.to_numpy()
    categories, counts = np.unique(x_idx, return_counts=True)
    k = len(categories)
    n = len(x_idx)

    if k == 1:
        return 1.0

    mu_x = np.sum(categories * counts) / n
    p_i = counts / n
    d_x = k - 1
    term = p_i * np.log2(1 - np.abs(categories - mu_x) / d_x)
    consensus = 1 + term.sum()
    return consensus

print(df.group_by("model").agg(pl.col("funniness")
                               .quantile(0.5).alias("funniness median"),
                               pl.col("funniness")
                               .map_batches(tastle_wierman_consensus)
                               .alias("funniness consensus").get(0),
                               pl.col("relation")
                               .quantile(0.5).alias("relation median"),
                               pl.col("relation")
                               .map_batches(tastle_wierman_consensus)
                               .alias("relation consensus").get(0))
      .sort("model")
      .write_csv())

# Funniness, Typicality, Relation, and Similarity correlation
print("###### Correlation analysis ######")
print(df.select(pl.col("funniness"),
                pl.col("typicality"),
                pl.col("relation"),
                pl.col("similarity")).corr())
print()

# Krippendorff's alpha
print("###### Agreement analysis ######")
models = df["model"].unique()
funniness_alpha, relation_alpha = list(), list()
for model in models:
    fun_data = (df.filter(pl.col("model") == model)
                .select("evaluator", "model")
                .join(df.select("headline_id"), how="cross").unique()
                .join(df, on=["evaluator", "model", "headline_id"], how="left")
                .drop_nulls()
                .pivot(index="evaluator", on="headline_id", values="funniness")
                .drop("evaluator"))
    fun_alpha = krippendorff.alpha(fun_data, level_of_measurement="ordinal")
    funniness_alpha.append(fun_alpha)

    rel_data = (df.filter(pl.col("model") == model)
                .select("evaluator", "model")
                .join(df.select("headline_id"), how="cross").unique()
                .join(df, on=["evaluator", "model", "headline_id"], how="left")
                .drop_nulls()
                .pivot(index="evaluator", on="headline_id", values="relation")
                .drop("evaluator"))
    rel_alpha = krippendorff.alpha(rel_data, level_of_measurement="ordinal")
    relation_alpha.append(rel_alpha)


alpha = pl.DataFrame({"model": models, "funniness alpha": funniness_alpha,
                      "relation alpha": relation_alpha})
print(alpha.write_csv())

# Llama3.3+shot Relation disagreements
print("###### Llama3.3+shot relation disagreements ######")
headlines = df["headline_id"].unique()
num_disagreements = 0
for headline_id in headlines:
    llama_shot = (df.filter(pl.col("model") == "Llama3.3+shot")
                  .filter(pl.col("headline_id") == headline_id))
    if llama_shot["relation"].n_unique() > 1:
        num_disagreements += 1
    if llama_shot["relation"].n_unique() > 2:
        headline = llama_shot["headline"].first()
        joke = llama_shot["generated"].first()
        ratings = llama_shot.select("evaluator", "relation").sort("evaluator")
        print(f"Headline: {headline}")
        print(f"Joke: {joke}")
        print(f"Ratings: {', '.join([rel_order[r - 1] for r in ratings['relation']])}")
        print()
print(f"Total disagreements in Llama3.3+shot: {num_disagreements}")

# Best jokes for every model
print("###### Best jokes for every model ######")
best_jokes = (agg_df.filter(pl.col("funniness") == 3)
              .group_by("model").agg(pl.col("headline_id").sample(1).get(0))
              .join(df, on=["model", "headline_id"])
              .select("model", "headline", "generated")
              .unique())
print(best_jokes.write_csv())

if not args.charts:
    exit()

# Funniness
sort_fun = df.group_by("model").mean().sort("funniness", descending=True)
fun_base = (alt.Chart(df, title=alt.Title("Funniness", dy=-10))
            .mark_bar(size=20)
            .encode(alt.X("count()")
                    .stack("normalize")
                    .axis(alt.Axis(domain=False,
                                   ticks=False,
                                   labels=False))
                    .scale(alt.Scale(domain=[0, 1], clamp=True, nice=True))
                    .title("Proportion of jokes (% of 108 = 27 headlines × 4 evaluators)"),
                    alt.Y("model:N")
                    .sort(sort_fun["model"].to_list())
                    .axis(alt.Axis(domain=False,
                                   ticks=False,
                                   labelPadding=10)))
            .properties(width=500, height=300))
fun_bars = (fun_base.mark_bar()
            .encode(color=alt.Color("funniness:O")
                    .scale(domain=[1, 2, 3], range=bar_color_scheme)
                    .legend(None)))
ratio_fun_text = (fun_base.transform_joinaggregate(count="count()",
                                                   groupby=["model", "funniness"])
                  .transform_joinaggregate(total="count()",
                                           groupby=["model"])
                  .transform_calculate(ratio="datum.count / datum.total")
                  .mark_text(size=7, align="right", dx=-3,
                             fontWeight="bold")
                  .encode(text=alt.Text("ratio:Q", format=".0%"),
                          color=alt.Color("funniness:O")
                          .scale(domain=[1, 2, 3], range=text_color_scheme)
                          .legend(None)))
left_fun_text = (fun_bars
                 .mark_text(size=11, align="left",
                            baseline="top", dy=-10)
                 .encode(x=alt.value(0), y=alt.value(0),
                         text=alt.value("Not funny"),
                         color=alt.value("black")))
middle_fun_text = (fun_bars
                   .mark_text(size=11, align="center",
                              baseline="top", dy=-10)
                   .encode(x=alt.value(alt.expr("width / 2")), y=alt.value(0),
                           text=alt.value("A bit funny"),
                           color=alt.value("black")))
right_fun_text = (fun_bars
                  .mark_text(size=11, align="right",
                             baseline="top", dy=-10)
                  .encode(x=alt.value("width"), y=alt.value(0),
                          text=alt.value("Funny"),
                          color=alt.value("black")))
fun_c = (alt.layer(fun_bars, left_fun_text, middle_fun_text, right_fun_text, ratio_fun_text)
         .configure_axis(grid=False)
         .configure_view(stroke=None)
         .resolve_scale(color="independent"))
fun_c.save("results/img/funniness_dist.png", ppi=300)

# Relation
sort_rel = df.group_by("model").mean().sort("relation", descending=True)
rel_base = (alt.Chart(df, title=alt.Title("Relation", dy=-10))
            .encode(alt.X("count()")
                    .stack("normalize")
                    .axis(alt.Axis(domain=False,
                                   ticks=False,
                                   labels=False))
                    .scale(alt.Scale(domain=[0, 1], clamp=True, nice=True))
                    .title("Proportion of jokes (% of 108 = 27 headlines × 4 evaluators)"),
                    alt.Y("model:N")
                    .sort(sort_rel["model"].to_list())
                    .axis(alt.Axis(domain=False,
                                   ticks=False,
                                   labelPadding=10)),
                    order="relation:O")
            .properties(width=500, height=300))
rel_bars = (rel_base.mark_bar()
            .encode(color=alt.Color("relation:O")
                    .scale(domain=[1, 2, 3], range=bar_color_scheme)
                    .legend(None)))
ratio_rel_text = (rel_base.transform_joinaggregate(count="count()",
                                                   groupby=["model", "relation"])
                  .transform_joinaggregate(total="count()",
                                           groupby=["model"])
                  .transform_calculate(ratio="datum.count / datum.total")
                  .mark_text(size=7, align="right", dx=-3,
                             fontWeight="bold")
                  .encode(text=alt.Text("ratio:Q", format=".0%"),
                          color=alt.Color("relation:O")
                          .scale(domain=[1, 2, 3], range=text_color_scheme)
                          .legend(None)))
left_rel_text = (rel_bars
                 .mark_text(size=11, align="left",
                            baseline="top", dy=-10)
                 .encode(x=alt.value(0), y=alt.value(0),
                         text=alt.value("Not related"),
                         color=alt.value("black")))
middle_rel_text = (rel_bars
                   .mark_text(size=11, align="left",
                              baseline="top", dx=15, dy=-10)
                   .encode(x=alt.value(alt.expr("width / 2")), y=alt.value(0),
                           text=alt.value("A bit related"),
                           color=alt.value("black")))
right_rel_text = (rel_bars
                  .mark_text(size=11, align="right",
                             baseline="top", dy=-10)
                  .encode(x=alt.value("width"), y=alt.value(0),
                          text=alt.value("Related"),
                          color=alt.value("black")))
rel_c = (alt.layer(rel_bars, left_rel_text, middle_rel_text, right_rel_text, ratio_rel_text)
         .configure_axis(grid=False)
         .configure_view(stroke=None)
         .resolve_scale(color="independent"))
rel_c.save("results/img/relation_dist.png", ppi=300)
