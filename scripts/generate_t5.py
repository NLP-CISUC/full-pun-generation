import polars as pl
from transformers import AutoTokenizer, T5ForConditionalGeneration
from argparse import ArgumentParser
from pathlib import Path
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

parser = ArgumentParser()
parser.add_argument("--input",
                    help="Input file path in JSONL format (processed headlines file)",
                    type=Path, required=True)
parser.add_argument("--definitions",
                    help="Include word definitions into the prompt",
                    action="store_true")
args = parser.parse_args()

df = pl.read_ndjson(args.input)
if args.definitions:
    prompts = df.select(
        pl.concat_str([
            pl.lit("Gerar trocadilho: "),
            pl.col("pun sign"),
            pl.lit(" ("),
            pl.col("pun definition"),
            pl.lit(") / "),
            pl.col("alternative sign"),
            pl.lit(" ("),
            pl.col("alternative definition"),
            pl.lit(")")
        ]))
else:
    prompts = df.select(
        pl.concat_str([
            pl.lit("Gerar trocadilho: "),
            pl.col("pun sign"),
            pl.lit(" / "),
            pl.col("alternative sign")
        ]))
prompts = prompts.to_series().fill_null("null").to_list()

tokenizer = AutoTokenizer.from_pretrained("unicamp-dl/ptt5-v2-base", legacy=True,
                                          device=device)
tokenizer.pad_token = tokenizer.eos_token
tokenized_prompts = tokenizer(prompts, truncation=True,
                              padding="max_length",
                              max_length=512,
                              return_tensors="pt")

model_subfolder = "ptt5-v2-descriptions" if args.definitions else "ptt5-v2-words"
model = T5ForConditionalGeneration.from_pretrained("Superar/ptt5-v2-pun-generation",
                                                   subfolder=model_subfolder,
                                                   device_map=device)
output = model.generate(input_ids=tokenized_prompts["input_ids"].to(device),
                        attention_mask=tokenized_prompts["attention_mask"].to(device),
                        max_new_tokens=512)
decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)
decoded_output = pl.Series("generated", decoded_output)

df = (df.with_columns(
    pl.concat_str([
        pl.lit("{\"palavras\":[\""),
        pl.col("pun sign"),
        pl.lit("\",\""),
        pl.col("alternative sign"),
        pl.lit("\"],\"trocadilho\":\""),
        decoded_output,
        pl.lit("\"}")])
    .alias("generated"))
)

savepath = Path("results/generation/ptt5-v2.jsonl")
if args.definitions:
    savepath = savepath.with_stem(savepath.stem + "_definitions")
savepath.parent.mkdir(exist_ok=True, parents=True)
df.write_ndjson(savepath)
