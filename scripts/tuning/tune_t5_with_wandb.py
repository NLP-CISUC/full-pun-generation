import json
import os
from argparse import ArgumentParser
from pathlib import Path

import polars as pl
import torch
import wandb
from datasets import Dataset
from full_pun_generation.puntuguese import Puntuguese
from transformers import (AutoTokenizer, DataCollatorForSeq2Seq,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          T5ForConditionalGeneration,
                          T5TokenizerFast, EarlyStoppingCallback)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--corpus", "-c",
                        help="Puntuguese corpus file path (puns.json)",
                        required=True, type=Path)
    parser.add_argument("--model_name", "-m",
                        help="HuggingFace model name to fine-tune",
                        required=False, type=str, default="unicamp-dl/ptt5-v2-base")
    parser.add_argument("--definitions", "-d",
                        help="Run prompt with pun and alternative signs definitions",
                        action="store_true")
    return parser.parse_args()


def load_tokenizer(model_name):
    model_name = args.model_name
    if Path(args.model_name).is_dir():
        with open(args.model_name + "/config.json") as json_file:
            config = json.load(json_file)
            model_name = config["_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_data(corpus_path, use_definitions=False):
    puntuguese = Puntuguese(corpus_path)
    puntuguese.filter_data()
    puntuguese.create_prompts(use_definitions)
    return puntuguese


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


def model_init(model_name):
    return T5ForConditionalGeneration.from_pretrained(model_name)


def wandb_hp_space(trial, use_definitions=False):
    name = "PunGeneration-T5-definitions" if use_definitions else "PunGeneration-T5-words"
    return {
        "method": "bayes",
        "name": name,
        "metric": {"name": "val_loss", "goal": "minimize"},
        "parameters": {
            "learning_rate": {"distribution": "uniform", "min": 1e-6, "max": 1e-3},
            "per_device_train_batch_size": {"values": [8, 16]},
        },
    }


def main(args):
    tokenizer = load_tokenizer(args.model_name)
    data = load_data(args.corpus, args.definitions)
    tokenized_datasets = {"train": tokenize_data(data.train, tokenizer),
                          "validation": tokenize_data(data.validation, tokenizer),
                          "test": tokenize_data(data.test, tokenizer)}

    os.environ["WANDB_PROJECT"] = "PunGeneration"
    wandb.login()
    training_args = Seq2SeqTrainingArguments(
        output_dir="results/models/ptt5-v2",
        overwrite_output_dir=True,
        num_train_epochs=500,
        learning_rate=1e-3,
        save_total_limit=1,
        save_steps=0.1,
        eval_strategy="steps",
        eval_steps=0.1,
        predict_with_generate=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb"
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    trainer = Seq2SeqTrainer(
        model=None,
        model_init=lambda: model_init(args.model_name),
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3,
                                         early_stopping_threshold=0.01)]
    )
    best_trial = trainer.hyperparameter_search(
        direction="minimize",
        backend="wandb",
        hp_space=lambda trial: wandb_hp_space(trial, args.definitions),
        n_trials=20
    )

if __name__ == '__main__':
    args = parse_args()
    main(args)
