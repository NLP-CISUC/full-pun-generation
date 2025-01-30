import polars as pl
from langchain_ollama import OllamaLLM
from full_pun_generation.puntuguese import Puntuguese

# puntuguese = Puntuguese("../../Resources/Corpora/BRHuM/data/puns.json")
# puntuguese.filter_data()
# puntuguese.prepare_prompts()
# homograph = puntuguese.train.filter(pl.col("homograph")).sample(n=1)
# homophone = puntuguese.train.filter(~pl.col("homograph")).sample(n=1)
# print(homograph)
# print(homophone)

model = OllamaLLM(base_url="",
                  model="llama3.3",
                  temperature=0.6, top_p=0.95)
few_shot_prompt = '''
[INSTRUÇÃO GERAL]
Você é um gerador de piadas baseado em trocadilhos em português.
Sua tarefa é criar piadas curtas e engraçadas baseadas nos pares de palavras fornecidos.
Crie os trocadilhos com base nos exemplos.
Apresente o resultado exclusivamente em formato JSON. Retorne apenas o objeto JSON.

[EXEMPLOS]
Palavras: "cinto" e "sinto"
Definições: "" e ""
{
    "palavras": ["cinto", "sinto"],
    "piada": "Por que o fantasma foi comprar um cinto? Para segurar as emoções quando eu o \\"sinto\\"!"
}

Palavras: "banco" e "banco"
Definições: "Assento onde se senta" e "Instituição financeira"
{
    "palavras": ["banco", "banco"],
    "piada": "Fui ao banco e sentei. Só depois percebi que não tinha dinheiro para sacar!"
}

Palavras: "manga" e "manga"
Definições: "Extremidade da camisa onde se põe os braços" e "fruto da mangueira"
'''

print(model.invoke(few_shot_prompt))
