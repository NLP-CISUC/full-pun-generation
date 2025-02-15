import re
from argparse import ArgumentParser
from pathlib import Path

import polars as pl
from langchain_ollama import OllamaLLM
from full_pun_generation.puntuguese import Puntuguese
from langchain_core.prompts import (
    FewShotChatMessagePromptTemplate, ChatPromptTemplate)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--ollama_url",
                        help="URL to Ollama API.",
                        required=True, type=str)
    parser.add_argument("--model",
                        help="Model to run on Ollama.",
                        required=False, type=str,
                        default="deepseek-r1:70b")
    parser.add_argument("--few_shot",
                        help="Run prompt with few-shot.",
                        action="store_true")
    parser.add_argument("--definitions",
                        help="Run prompt with pun and alternative signs definitions",
                        action="store_true")
    return parser.parse_args()


def create_prompt(few_shot=False, include_definition=False):
    sys_prompt = """Você é um gerador de piadas baseado em trocadilhos em português.
    Sua tarefa é criar piadas curtas e engraçadas baseadas nos pares de palavras fornecidos.
    Crie os trocadilhos com base nos exemplos.
    Apresente o resultado exclusivamente em formato JSON contento duas chaves: \"palavras\" e \"trocadilho\", contendo as palavras fornecidas e a piada criada, respectivamente.
    Retorne apenas o objeto JSON."""

    human_prompt = "Palavras: \"{pun_sign}\" e \"{alt_sign}\""
    if include_definition:
        human_prompt += "\nDefinições: {definition}"

    example_prompt = ChatPromptTemplate.from_messages(
        [("human", human_prompt),
         ("ai", "{{\"palavras\": [\"{pun_sign}\", \"{alt_sign}\"], \"trocadilho\": \"{joke}\"}}")]
    )
    examples = [
        {"pun_sign": "entre-meada", "alt_sign": "entre miada",
         "joke": "O que é que está entre dois gatos? Uma entre-meada.",
         "definition": "\"Part of the pork meat consisting of bacon with bacon in between\" e \"Between the voice of the cat\""},
        {"pun_sign": "flora", "alt_sign": "flora",
         "joke": "Foi lançada uma nova manteiga com sabor a merda. Chama-se Flora Intestinal.",
         "definition": "\"Set of plants from a region, an environment or a geological period\" e \"Butter brand\""},
        {"pun_sign": "shanti-lee", "alt_sign": "chantilly",
         "joke": "Que nome se dá a uma chinesa muito docinha? Shanti-Lee.",
         "definition": "\"Chinese-sounding name\" e \"Cream made from whipped cream and sugar\""},
        {"pun_sign": "paciente", "alt_sign": "paciente",
         "joke": "O que o médico disse para o homem que entrou gemendo de dor no hospital? \"Seja paciente\".",
         "definition": "\"Who or what has patience\" e \"Any person undergoing medical treatment or care\""}
    ]

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        examples=examples,
        example_prompt=example_prompt
    )

    messages = [("system", sys_prompt)]
    if few_shot:
        messages.append(few_shot_prompt)
    messages.append(("human", human_prompt))
    final_prompt = ChatPromptTemplate.from_messages(messages)
    return final_prompt


def main(args):
    model = OllamaLLM(base_url=args.ollama_url,
                      model=args.model,
                      temperature=0.6, top_p=0.95)
    prompt = create_prompt(few_shot=args.few_shot,
                           include_definition=args.definitions)
    chain = prompt | model
    print(f"Prompt:\n{prompt}")

    def generate(row):
        prompt_data = {"pun_sign": row["pun sign"],
                       "alt_sign": row["alternative sign"]}
        if args.definitions:
            prompt_data["definition"] = f"\"{row['pun definition']}\" e \"{row['alternative definition']}\""
        return chain.invoke(prompt_data)

    model_name = re.sub(r"[:.]", "-", args.model)
    savepath = Path(f"results/generation/{model_name}.jsonl")
    if args.few_shot:
        savepath = savepath.with_stem(savepath.stem + "_fewshot")
    if args.definitions:
        savepath = savepath.with_stem(savepath.stem + "_definitions")
    savepath.parent.mkdir(exist_ok=True, parents=True)

    df = (pl.read_ndjson("data/processed_headlines.jsonl")
            .with_columns(
                pl.struct(pl.col("pun sign"),
                          pl.col("alternative sign"),
                          pl.col("pun definition"),
                          pl.col("alternative definition"))
                .map_elements(generate, return_dtype=pl.String)
                .alias("generated"))
          )
    df.write_ndjson(savepath)
    print(f"Saved {savepath}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
