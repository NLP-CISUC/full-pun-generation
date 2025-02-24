from argparse import ArgumentParser
from pathlib import Path

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
              .join(df.select(["headline_id", "model", "typicality", "similarity"]).unique(),
                    on=["headline_id", "model"], how="left")
              .select(pl.col("evaluator"),
                      pl.col("model"),
                      pl.col("headline_id"),
                      pl.col("funniness"),
                      pl.col("typicality"),
                      pl.col("relation"),
                      pl.col("similarity")))
print(f"Added {missing_df.height} missing rows.\n")
df = df.extend(missing_df)

# Descriptive statistics
df = (df.with_columns([pl.col("funniness").cast(pl.Enum(fun_order)).to_physical(),
                       pl.col("relation").cast(pl.Enum(rel_order)).to_physical()]))
agg_df = (df.group_by(["model", "headline_id"])
          .agg(pl.col("funniness").to_physical().mode().get(0),
               pl.col("typicality").first(),
               pl.col("relation").to_physical().mode().get(0),
               pl.col("similarity").first()))
agg_df = df

sort_fun = df.group_by("model").mean().sort("funniness", descending=True)
sort_rel = df.group_by("model").mean().sort("relation", descending=True)

# Funniness, Typicality, Relation, and Similarity correlation
print("###### Correlation analysis ######")
print(agg_df.select(pl.col("funniness"),
                    pl.col("typicality"),
                    pl.col("relation"),
                    pl.col("similarity")).corr())
print()

print("###### Agreement analysis ######")
# Krippendorff's alpha for funniness
fun_data = (df.select("evaluator")
            .join(df.select("model", "headline_id"), how="cross").unique()
            .join(df, on=["evaluator", "model", "headline_id"], how="left")
            .pivot(index="evaluator", on=["model", "headline_id"], values="funniness")
            .drop("evaluator")
            .to_numpy())
fun_alpha = krippendorff.alpha(fun_data, level_of_measurement="ordinal")
print(f"Krippendorff's alpha for funniness: {fun_alpha:.2f}")

# Krippendorff's alpha for relation
rel_data = (df.select("evaluator")
            .join(df.select("model", "headline_id"), how="cross").unique()
            .join(df, on=["evaluator", "model", "headline_id"], how="left")
            .pivot(index="evaluator", on=["model", "headline_id"], values="relation")
            .drop("evaluator")
            .to_numpy())
rel_alpha = krippendorff.alpha(rel_data, level_of_measurement="ordinal")
print(f"Krippendorff's alpha for relation: {rel_alpha:.2f}")

if not args.charts:
    exit()

# Funniness
fun_base = (alt.Chart(agg_df, title=alt.Title("Funniness", dy=-10))
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
                    .scale(domain=[0, 1, 2], range=bar_color_scheme)
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
                          .scale(domain=[0, 1, 2], range=text_color_scheme)
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
rel_base = (alt.Chart(agg_df, title=alt.Title("Relation", dy=-10))
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
                    .scale(domain=[0, 1, 2], range=bar_color_scheme)
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
                          .scale(domain=[0, 1, 2], range=text_color_scheme)
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
