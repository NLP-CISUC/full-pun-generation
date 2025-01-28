import json
from argparse import ArgumentParser
from pathlib import Path

import polars as pl
import torch
from datasets import Dataset, load_dataset
from full_pun_generation.wordnet import get_definitions_similarity
from nltk.corpus import wordnet as wn
from transformers import (AutoTokenizer, DataCollatorForLanguageModeling,
                          DataCollatorForSeq2Seq, GPTNeoForCausalLM,
                          GPT2TokenizerFast, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, T5ForConditionalGeneration,
                          T5TokenizerFast, Trainer, TrainingArguments)

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
    default="unicamp-dl/ptt5-v2-base",
)
parser.add_argument(
    "--output",
    "-o",
    help="File path to save predictions",
    required=False,
    type=Path,
    default=None
)
parser.add_argument(
    "--no_train",
    action="store_true",
    help="Do not train the model",
    required=False,
)
parser.add_argument(
    "--no_test",
    action="store_true",
    help="Do not test the model",
    required=False,
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
    # Only one alternative sign
    .filter(pl.col("alternative sign").list.len() == 1)
    .with_columns(pl.col("alternative sign").list.get(0))
    # Either homograph or homophone
    .filter(pl.col("homograph") | pl.col("homophone"))
)

# Substitute homographs by their definitions from WordNet
task_preffix = "Criar trocadilho"
inputs = (
    puntuguese.with_columns(
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
        .alias("updated signs")).select(
        pl.col("id"),
        pl.col("text"),
        pl.col("updated signs").struct.field("pun sign"),
        pl.col("updated signs").struct.field("alternative sign"))
    .filter(pl.col("pun sign") != "")
    .select(
        [
            pl.col("id"),
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
)

# Load tokenizer
model_path = Path(args.model_name)
model_name = args.model_name
if model_path.is_dir():
    json_file = open(args.model_name + "/config.json")
    config = json.load(json_file)
    json_file.close()
    tokenizer = AutoTokenizer.from_pretrained(config["_name_or_path"])
    model_name = config["_name_or_path"]
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
tokenizer.pad_token = tokenizer.eos_token

# Will use model_type to get the correct preprocessing, model, data_collator, etc.
model_type = ""
if isinstance(tokenizer, T5TokenizerFast):
    model_type = "t5"
if isinstance(tokenizer, GPT2TokenizerFast):
    model_type = "gptneo"

# Split and tokenize
hf_puntuguese = load_dataset("Superar/Puntuguese")
splits = {"train": [id_[:-2] for id_ in hf_puntuguese["train"]["id"] if id_.endswith("H")],
          "eval": [id_[:-2] for id_ in hf_puntuguese["validation"]["id"] if id_.endswith("H")],
          "test": [id_[:-2] for id_ in hf_puntuguese["test"]["id"] if id_.endswith("H")]}
datasets = dict()
for split, split_ids in splits.items():
    split_inputs = inputs.filter(pl.col("id").is_in(split_ids))

    if model_type == "gptneo":
        if split == "test":
            split_inputs = split_inputs.with_columns(
                    pl.concat_str(
                        [pl.col("command"),
                         pl.lit(" ### ")],
                        separator="",
                    )
                )
        else:
            split_inputs = split_inputs.with_columns(
                    pl.concat_str(
                        [pl.col("command"),
                         pl.col("label")],
                        separator=" ### ",
                    )
                )


    tokenized_split = tokenizer(
        split_inputs["command"].to_list(),
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )
    tokenized_split["labels"] = tokenizer(
        split_inputs["label"].to_list(),
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )["input_ids"]
    datasets[split] = Dataset.from_dict(tokenized_split)

# Deal with different types of models
model = None
data_collator_class = None
data_collator_args = {"tokenizer": tokenizer}
training_args_class = None
training_args_args = {"output_dir": "results/models", "overwrite_output_dir": True,
                      "num_train_epochs": 200, "learning_rate": 1e-3,
                      "save_total_limit": 1, "eval_steps": 0.1}
trainer_class = None
if model_type == "t5":
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    data_collator_class = DataCollatorForSeq2Seq
    data_collator_args["model"] = model
    training_args_class = Seq2SeqTrainingArguments
    training_args_args["predict_with_generate"] = True
    trainer_class = Seq2SeqTrainer
if model_type == "gptneo":
    model = GPTNeoForCausalLM.from_pretrained(args.model_name)
    data_collator_class = DataCollatorForLanguageModeling
    data_collator_args["mlm"] = False
    training_args_class = TrainingArguments
    training_args_args["per_device_train_batch_size"] = 4
    trainer_class = Trainer

# Training
if not args.no_train and model_type:
    data_collator = data_collator_class(**data_collator_args)
    training_args = training_args_class(**training_args_args)
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["eval"],
        data_collator=data_collator,
    )
    trainer.train()

# Test
if not args.no_test and model:
    split_ids = splits["test"]
    test_df = (inputs.filter(pl.col("id").is_in(split_ids))
               .join(puntuguese, on=pl.col("id"), how="left"))
    with torch.no_grad():
        input_ids = torch.tensor(datasets["test"]["input_ids"])
        attention_mask = torch.tensor(datasets["test"]["attention_mask"])
        output = model.generate(input_ids=input_ids,
                                attention_mask=attention_mask,
                                do_sample=True,
                                temperature=1.0,
                                num_beams=4,
                                top_p=0.8,
                                repetition_penalty=1.2,
                                max_new_tokens=512)
        test_df = test_df.with_columns(pl.Series(
            "prediction", tokenizer.batch_decode(output, skip_special_tokens=True)))
    test_df = test_df.select(
        pl.col(["id", "homograph", "homophone", "pun sign",
               "alternative sign", "label", "prediction"])
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        test_df.write_ndjson(args.output)
    else:
        print(test_df)
