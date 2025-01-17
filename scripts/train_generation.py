from argparse import ArgumentParser
from pathlib import Path

import polars as pl
from full_pun_generation.wordnet import get_definitions_similarity
from nltk.corpus import wordnet as wn
from transformers import (
    DataCollatorForLanguageModeling,
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
)

parser = ArgumentParser()
parser.add_argument(
    "--corpus",
    "-c",
    help="Puntuguese corpus file path (puns.json)",
    required=True,
    type=Path,
)
parser.add_argument(
    "--model_name",
    "-m",
    help="HuggingFace model name to fine-tune",
    required=False,
    type=str,
    default="unicamp-dl/ptt5-base-portuguese-vocab",
)
args = parser.parse_args()


def update_signs(row):
    if not row["homograph"]:
        return {
            "pun sign": row["pun sign"],
            "alternative sign": row["alternative sign"],
        }
    synsets = wn.synsets(row["pun sign"], lang="por")
    if len(synsets) < 2:
        return {"pun sign": "", "alternative sign": ""}
    _, d1, d2 = get_definitions_similarity(synsets)
    return {"pun sign": d1, "alternative sign": d2}


# Read and preprocess the Puntuguese corpus
puntuguese = pl.read_json(args.corpus)
puntuguese = (
    puntuguese.filter(pl.col("signs").list.len() == 1)  # Only one sign
    .with_columns(pl.col("signs").list.get(0))  # Deal with dictionaries
    .unnest("signs")
    .filter(pl.col("alternative sign").list.len() == 1)  # Only one alternative sign
    .with_columns(pl.col("alternative sign").list.get(0))
    .filter(pl.col("homograph") | pl.col("homophone"))  # Either homograph or homophone
    .with_columns(  # Substitute homographs by their definitions from WordNet
        pl.struct(["pun sign", "alternative sign", "homograph"])
        .map_elements(
            update_signs,
            return_dtype=pl.Struct(
                [
                    pl.Field("pun sign", pl.String),
                    pl.Field("alternative sign", pl.String),
                ]
            ),
        )
        .alias("updated signs")
    )
    .select(
        pl.col("text"),
        pl.col("updated signs").struct.field("pun sign"),
        pl.col("updated signs").struct.field("alternative sign"),
    )
    .filter(pl.col("pun sign") != "")  # Ignore homographs that are not in WordNet
)

task_preffix = "Criar trocadilho"
inputs = puntuguese.select(
    [
        pl.concat_str(
            [
                pl.lit(task_preffix),
                pl.concat_str(
                    [pl.col("pun sign"), pl.col("alternative sign")], separator="/"
                ),
            ],
            separator=": ",
        ).alias("command"),
        pl.col("text").alias("label"),
    ]
)

tokenizer = T5Tokenizer.from_pretrained(args.model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenized_input = tokenizer(
    inputs["command"].to_list(),
    truncation=True,
    padding="max_length",
    max_length=128,
)
tokenized_labels = tokenizer(
    inputs["label"].to_list(), truncation=True, padding="max_length", max_length=128
)
tokenized_input["labels"] = tokenized_labels["input_ids"]

model = T5ForConditionalGeneration.from_pretrained(args.model_name)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
training_args = TrainingArguments(
    output_dir="results/models",
    overwrite_output_dir=True,
    num_train_epochs=4,
    learning_rate=1e-4,
    save_total_limit=2,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_input,
    data_collator=data_collator,
)
trainer.train()
