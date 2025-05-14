# A Full Pipeline for Context-Aware Pun Generation

This paper proposes a full pipeline for Pun Generation, including the creation of the pun and alternative word pairs from a given context. Our experiments were carried out in Portuguese.

## How to install

This project uses Python 3.9.

If you are using uv, run the following command:

```bash
uv init --python=3.9
uv sync
```

If you are using pip:

```bash
pip install -r requirements.txt
```

Afterward, remember to install the NLTK WordNet corpus:

```python
import nltk
nltk.download('omw-1.4')
```

## How to run

The pipeline's implementation (keyword extraction and expansion, homograph and homophone identification) is in the `src` folder. Our experiments, including NLG methods, results analysis, and other toy projects are in the `scripts` folder. The most important scripts are:

- `scripts/generation/generate_ollama_jokes.py`: Generate jokes using Ollama LLMs
- `scripts/generation/generate_t5_jokes.py`: Fine-tune and generate jokes using T5

All scripts require the data to be in the `data` folder in the `data/processed_headlines.jsonl` file, which already includes all pun and alternative signs created, in JSONL format. To create this file, you can use the `scripts/preprocessing/preprocess_headlines.py` script.

Generation results are saved in the `results/generation/` folder, separated by the generation method.

### Evaluation interface

The evaluation interface implementation is in the `evaluation_interface` folder, which requires [streamlit](https://streamlit.io/) to run. All evaluation results are in the `results/evaluation/` folder, separated by evaluator. More information on how to configure the evaluation interface can be found in its own README file.

## How to cite

```bibtex
@inproceedings{InacioMarcioLimaGoncaloOliveira2024,
    title     = {A {Full Pipeline} for {Context-Aware Pun Generation}},
    booktitle = {Proceedings of the 16th {{International Conference}} on {{Computational Creativity}} ({{ICCC}}'25)},
    author    = {In{\'a}cio, Marcio Lima and Gon{\c c}alo Oliveira, Hugo},
    year      = {2025},
    month     = jun,
    publisher = {Association for Computational Creativity (ACC)},
    address   = {Campinas}
}
```

> Inácio, M. L., & Gonçalo Oliveira, H. (2025, June). A Full Pipeline for Context-Aware Pun Generation. Proceedings of the 16th International Conference on Computational Creativity (ICCC’25). 16th International Conference on Computational Creativity (ICCC’25), Campinas.
