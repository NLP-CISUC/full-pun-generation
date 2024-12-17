import logging

import gradio as gr

from .context import extract_keywords, expand_keywords

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.StreamHandler()])

def format_keywords(text, n_keywords=5):
    keywords = extract_keywords(text, n_keywords)
    expanded_keywords = expand_keywords(keywords)
    return '\n'.join([str(kw) for kw, _ in expanded_keywords])

def create_interface():
    iface = gr.Interface(
            fn=format_keywords,
            inputs=[gr.Textbox(lines=10, label='Input text'),
                    gr.Number(value=5, label='Number of keywords')],
            outputs=gr.Textbox(label='Keywords'),
            title='Pun Generation',
            description='Generate pun based on input text',
            theme='soft')
    return iface

def main():
    iface = create_interface()
    iface.launch()

if __name__ == '__main__':
    main()

