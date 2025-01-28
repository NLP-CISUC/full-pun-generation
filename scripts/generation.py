import json
from argparse import ArgumentParser
from pathlib import Path

import polars as pl
import torch
from datasets import Dataset, load_dataset
from full_pun_generation.wordnet import get_definitions_similarity
from nltk.corpus import wordnet as wn
from transformers import (AutoTokenizer, DataCollatorForLanguageModeling,
                          DataCollatorForSeq2Seq, GPT2TokenizerFast,
                          GPTNeoForCausalLM, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, T5ForConditionalGeneration,
                          T5TokenizerFast, Trainer, TrainingArguments)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--corpus", "-c",
                        help="Puntuguese corpus file path (puns.json)",
                        required=True, type=Path)
    parser.add_argument("--model_name", "-m",
                        help="HuggingFace model name to fine-tune",
                        required=False, type=str, default="unicamp-dl/ptt5-v2-base")
    parser.add_argument("--output", "-o",
                        help="File path to save predictions",
                        required=False, type=Path, default=None)
    parser.add_argument("--no_train", action="store_true",
                        help="Do not train the model", required=False)
    parser.add_argument("--no_test", action="store_true",
                        help="Do not test the model", required=False)
    return parser.parse_args()


def load_data(corpus_path):
    def update_signs(row):
        if not row["homograph"]:
            return row
        synsets = wn.synsets(row["pun sign"], lang="por")
        if len(synsets) < 2:
            return {"pun sign": "", "alternative sign": ""}
        _, d1, d2 = get_definitions_similarity(synsets)
        return {"pun sign": d1, "alternative sign": d2}
    puntuguese = pl.read_json(corpus_path)
    return (
        puntuguese.filter(pl.col("signs").list.len() == 1)  # Only one sign
        .with_columns(pl.col("signs").list.get(0))  # Deal with dictionaries
        .unnest("signs")
        # Only one alternative sign
        .filter(pl.col("alternative sign").list.len() == 1)
        .with_columns(pl.col("alternative sign").list.get(0))
        # Either homograph or homophone
        .filter(pl.col("homograph") | pl.col("homophone"))
        .with_columns(
            pl.struct(["pun sign", "alternative sign", "homograph"])
            .map_elements(update_signs)
            .alias("updated signs")
        )
        .select(
            pl.col("id"),
            pl.col("text"),
            pl.col("updated signs").struct.field("pun sign"),
            pl.col("updated signs").struct.field("alternative sign")
        )
        .filter(pl.col("pun sign") != "")
        .with_columns(
            pl.concat_str([
                pl.lit("Criar trocadilho: "),
                pl.concat_str([pl.col("pun sign"), pl.col(
                    "alternative sign")], separator="/")
            ]).alias("command")
        )
    )


def load_tokenizer(model_name):
    model_name = args.model_name
    if Path(args.model_name).is_dir():
        with open(args.model_name + "/config.json") as json_file:
            config = json.load(json_file)
            model_name = config["_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_data(inputs, tokenizer, max_length=512):
    tokenized = tokenizer(
        inputs["command"].to_list(),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    tokenized["labels"] = tokenizer(
        inputs["text"].to_list(),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )["input_ids"]
    return Dataset.from_dict(tokenized)


def setup_training(model_type, model_name, tokenizer):
    if model_type == "t5":
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        data_collator = DataCollatorForSeq2Seq(
            model=model, tokenizer=tokenizer)
        training_args = Seq2SeqTrainingArguments(
            output_dir="results/models",
            overwrite_output_dir=True,
            num_train_epochs=200,
            learning_rate=1e-3,
            save_total_limit=1,
            eval_steps=0.1,
            predict_with_generate=True
        )
        trainer_class = Seq2SeqTrainer
    elif model_type == "gptneo":
        model = GPTNeoForCausalLM.from_pretrained(model_name)
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False)
        training_args = TrainingArguments(
            output_dir="results/models",
            overwrite_output_dir=True,
            per_device_train_batch_size=4,
            num_train_epochs=200,
            learning_rate=1e-3,
            save_total_limit=1,
            eval_steps=0.1
        )
        trainer_class = Trainer
    else:
        raise ValueError("Model type not supported")
    return model, data_collator, training_args, trainer_class


def preprocess_causal_data(data, is_test=False):
    data = data.with_columns(
        pl.concat_str([
            pl.col("command"),
            pl.lit(" ### ")
        ], separator="")
    )
    if not is_test:
        data = data.with_columns(
            pl.concat_str([
                pl.col("command"),
                pl.col("text"),
            ], separator="")
        )
    return data


def main(args):
    data = load_data(args.corpus)
    tokenizer = load_tokenizer(args.model_name)

    # Will use model_type to get the correct preprocessing, model, data_collator, etc.
    model_type = ""
    if isinstance(tokenizer, T5TokenizerFast):
        model_type = "t5"
    elif isinstance(tokenizer, GPT2TokenizerFast):
        model_type = "gptneo"

    model, data_collator, training_args, trainer_class = setup_training(
        model_type, args.model_name, tokenizer)

    # Split and tokenize
    hf_puntuguese = load_dataset("Superar/Puntuguese")
    splits = {split: [id_[:-2] for id_ in hf_puntuguese[split]["id"]
                      if id_.endswith("H")] for split in hf_puntuguese}
    datasets = dict()
    for split, split_ids in splits.items():
        split_df = data.filter(pl.col("id").is_in(split_ids))
        is_test = (split in ["test", "validation"])
        datasets[split] = preprocess_causal_data(split_df, is_test)
    tokenized_datasets = {split: tokenize_data(df, tokenizer)
                          for split, df in datasets.items()}

    # Training
    if not args.no_train:
        trainer = trainer_class(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
        )
        trainer.train()

    # Test
    if not args.no_test:
        test_df = datasets["test"]
        with torch.no_grad():
            input_ids = torch.tensor(tokenized_datasets["test"]["input_ids"])
            attention_mask = torch.tensor(
                tokenized_datasets["test"]["attention_mask"])
            output = model.generate(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    do_sample=True,
                                    temperature=1.0,
                                    num_beams=4,
                                    top_p=0.8,
                                    repetition_penalty=1.2,
                                    max_new_tokens=512)
            test_df = test_df.with_columns(
                pl.Series("prediction",
                          tokenizer.batch_decode(output, skip_special_tokens=True))
            )

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            test_df.write_ndjson(args.output)
        else:
            print(test_df)


if __name__ == '__main__':
    args = parse_args()
    main(args)
