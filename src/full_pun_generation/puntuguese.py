import polars as pl
from datasets import load_dataset
from nltk.corpus import wordnet as wn

from full_pun_generation.wordnet import (get_definitions_similarity,
                                         get_words_synsets,
                                         get_valid_words)


class Puntuguese():
    def __init__(self, filepath):
        self.data = pl.read_json(filepath)
        self.splits = self._get_splits()

    @property
    def train(self):
        return self.data.filter(pl.col("id").is_in(self.splits["train"]))

    @property
    def validation(self):
        return self.data.filter(pl.col("id").is_in(self.splits["validation"]))

    @property
    def test(self):
        return self.data.filter(pl.col("id").is_in(self.splits["test"]))

    def _get_splits(self):
        hf_dataset = load_dataset("Superar/Puntuguese")
        splits = {split: [id_[:-2] for id_ in hf_dataset[split]["id"]
                          if id_.endswith("H")] for split in hf_dataset}
        return splits

    def filter_data(self):
        """
        Only include homograph and homophones with exactly one pun
        and one alternative signs.
        """
        self.data = (
            self.data.filter(pl.col("signs").list.len() == 1)
            .with_columns(pl.col("signs").list.get(0))
                .unnest("signs")
                .filter(pl.col("alternative sign").list.len() == 1)
                .with_columns(pl.col("alternative sign").list.get(0))
                .filter(pl.col("homograph") | pl.col("homophone"))
        )

    def include_definitions(self):
        def get_definitions(row):
            valid_signs = get_valid_words(
                [row["pun sign"], row["alternative sign"]])
            if len(valid_signs) < 2:
                return ["", ""]
            synsets1, synsets2 = get_words_synsets([row["pun sign"],
                                                    row["alternative sign"]])
            _, def1, def2 = get_definitions_similarity(synsets1, synsets2)
            return [def1, def2]
        if "pun sign" not in self.data.columns:
            raise ValueError("Data must be filtered first.")

        self.data = (self.data
                     .with_columns(
                         pl.struct(
                             ["pun sign", "alternative sign", "homograph"])
                         .map_elements(get_definitions, return_dtype=pl.List(pl.String))
                         .alias("definitions"))
                     .with_columns(
                         pl.col("definitions").list.get(
                             0).alias("pun definition"),
                         pl.col("definitions").list.get(1).alias("alternative definition"))
                     .drop("definitions")
                     )

    def create_prompts(self, use_definitions=False):
        if use_definitions:
            if "pun definition" not in self.data.columns:
                self.include_definitions()
            self.data = (self.data
                         .filter(
                             (pl.col("pun definition") != "") &
                             (pl.col("alternative definition") != "")
                         )
                         .with_columns(
                             pl.concat_str([
                                 pl.lit("Gerar trocadilho: "),
                                 pl.col("pun sign"),
                                 pl.lit(" ("),
                                 pl.col("pun definition"),
                                 pl.lit(") / "),
                                 pl.col("alternative sign"),
                                 pl.lit(" ("),
                                 pl.col("alternative definition"),
                                 pl.lit(")")])
                             .alias("command"))
                         )
        else:
            self.data = (self.data
                         .with_columns(
                             pl.concat_str([
                                 pl.lit("Gerar trocadilho: "),
                                 pl.col("pun sign"),
                                 pl.lit(" / "),
                                 pl.col("alternative sign")
                             ]).alias("command"))
                         )


if __name__ == "__main__":
    puntuguese = Puntuguese("../Puntuguese/data/puns.json")
    puntuguese.filter_data()
    print(puntuguese.data)
    puntuguese.create_prompts()
    print(puntuguese.data)
    puntuguese.create_prompts(use_definitions=True)
    print(puntuguese.data)
    print(puntuguese.train.height)
    print(puntuguese.validation.height)
    print(puntuguese.test.height)
