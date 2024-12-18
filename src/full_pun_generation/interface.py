import logging

import gradio as gr

from .context import extract_keywords, expand_keywords
from .wordnet import get_ambiguous_words

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.StreamHandler()])


def get_signs(text, n_keywords=5):
    keywords = extract_keywords(text, n_keywords)
    expanded_keywords = expand_keywords(keywords)
    words = {kw for kw, _ in expanded_keywords}

    homographic_signs = get_ambiguous_words(words)
    return '\n'.join([str(w) for w, _ in homographic_signs])

def create_interface():
    iface = gr.Interface(
            fn=get_signs,
            inputs=[gr.Textbox(lines=10, label='Input text'),
                    gr.Number(value=5, label='Number of keywords')],
            outputs=gr.Textbox(label='Homographic signs'),
            title='Pun Generation',
            description='Generate pun based on input text',
            theme='huggingface')
    return iface

def main():
    iface = create_interface()
    iface.launch()

if __name__ == '__main__':
    main()

