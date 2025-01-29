import polars as pl
from datasets import load_dataset
from nltk.corpus import wordnet as wn

from full_pun_generation.wordnet import get_definitions_similarity


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

    def prepare_prompts(self):
        """
        Create columns with generation prompts.
        """
        def update_signs(row):
            if not row["homograph"]:
                return row
            synsets = wn.synsets(row["pun sign"], lang="por")
            if len(synsets) < 2:
                return {"pun sign": "", "alternative sign": ""}
            _, d1, d2 = get_definitions_similarity(synsets)
            return {"pun sign": d1, "alternative sign": d2}
        self.data = (self.data
                     .with_columns(
                         pl.struct(["pun sign", "alternative sign", "homograph"])
                         .map_elements(update_signs,
                                       return_dtype=pl.Struct(
                                           [pl.Field("pun sign", pl.String),
                                            pl.Field("alternative sign", pl.String)]
                                           )
                                       )
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
                         ))

    def prepare_causal_prompts(self):
        def truncate(row):
            is_test = (row["id"] in self.splits["test"])
            is_validation = (row["id"] in self.splits["validation"])
            if is_test or is_validation:
                return row["command"].split(" ### ")[0] + " ### "
            return row["command"]
        if "command" not in self.data.columns:
            self.prepare_prompts()
        self.data = (self.data
                     .with_columns(pl.concat_str([pl.col("command"), pl.col("text")],
                                                 separator=" ### "))
                     .with_columns(pl.struct(["command", "id"])
                                   .map_elements(truncate)))

if __name__ == "__main__":
    puntuguese = Puntuguese("../../Resources/Corpora/BRHuM/data/puns.json")
    puntuguese.filter_data()
    puntuguese.prepare_prompts()
    puntuguese.prepare_causal_prompts()
    print(puntuguese.data)
