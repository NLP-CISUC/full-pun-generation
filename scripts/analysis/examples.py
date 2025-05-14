from pathlib import Path

import polars as pl
import yaml

fun_order = ["Não tem piada", "Tem pouca piada", "Tem piada"]
rel_order = ["Não tem relação", "Tem pouca relação", "Tem relação"]
model_names = {"deepseek-r1-70b": "Deepseek",
               "llama3-3": "Llama3.3",
               "ptt5-v2": "PTT5",
               "_definitions": "+def",
               "_fewshot": "+shot"}

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

example_headlines = ["Starlink Direct to Cell passa a disponibilizar 4G via satélite para todos os smartphones",
                     "Starlink Direct to Cell passa a disponibilizar 4G via satélite para todos os smartphones",
                     "Vodafone realiza primeira videochamada espacial do mundo com smartphone",
                     "\"Não daria à minha filha um vibrador aos 12 anos. Não posso é condenar\"",
                     "Liga Europa: o onze provável do FC Porto na estreia de Martín Anselmi",
                     "O preço do Samsung Galaxy S25 cai a pique com os descontos na Samsung",
                     "Restos de salmoura no asteroide Bennu contêm minerais essenciais para a vida.",
                     "E assim foi o terceiro episódio de \"A Herança\"",
                     "Restos de salmoura no asteroide Bennu contêm minerais essenciais para a vida.",
                     "Restos de salmoura no asteroide Bennu contêm minerais essenciais para a vida."]
examples_jokes = ["Qual é o satélite que mais se aproxima do homem? O satélite.",
                  "Por que o homem não tem um satélite? Porque não tem um satélite.",
                  "Por que o mundo foi ao psicólogo? Porque estava se sentindo um pouco 'girado'!",
                  "Eu condeno o uso excessivo de 'condenar', mas não posso condenar alguém por concordar!",
                  "Por que a lâmpada foi ao terapeuta? Ela estava com problemas para se ligar.",
                  "Por que o vendedor usava uma venda nos olhos? Para não ver a venda cair!",
                  "O asteróide foi ao terapeuta porque estava se sentindo um pouco 'defasado'.",
                  "Por que a carteira e o DNA foram juntos ao terapeuta? Porque um carrega a herança material e o outro, a herança genética!",
                  "O que é um asteróide? É um asteroide sem sotaque.",
                  "Por que o asteroide foi ao festival de música? Porque ele amava rock e metal!"]
examples_models = ["PTT5",
                   "PTT5+def",
                   "Llama3.3",
                   "Llama3.3+def",
                   "Llama3.3+shot",
                   "Llama3.3+shot+def",
                   "Deepseek",
                   "Deepseek+def",
                   "Deepseek+shot",
                   "Deepseek+shot+def"]
examples_df = pl.DataFrame({"headline": example_headlines,
                            "generated": examples_jokes,
                            "model": examples_models})
print(examples_df.join(df, on=["headline", "generated", "model"], how="left")
      .group_by("model", "headline_id")
      .agg(pl.col("funniness").mode().max(),
           pl.col("typicality").first(),
           pl.col("relation").mode().max(),
           pl.col("similarity").first()).write_csv())



