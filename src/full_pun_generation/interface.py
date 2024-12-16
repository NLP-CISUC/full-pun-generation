import logging

import gradio as gr
from keybert import KeyBERT
from transformers import pipeline

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.StreamHandler()])

kw_model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')
pos_model = pipeline('ner', model='Emanuel/porttagger-base')

def pos_tagging(text):
    logging.info('Performing POS tagging')
    doc = pos_model(text)
    pos_tags = [(ent['word'], ent['entity']) for ent in doc]
    logging.info(f'POS tags: {pos_tags}')

    # Deal with subword tokens that start with '##'
    merged_tags = []
    i = 0
    while i < len(pos_tags):
        if not pos_tags[i][0].startswith('##'):
            current_word = pos_tags[i][0]
            current_tag = pos_tags[i][1]

            while i + 1 < len(pos_tags) and pos_tags[i + 1][0].startswith('##'):
                current_word += pos_tags[i + 1][0][2:]
                i += 1
            merged_tags.append((current_word, current_tag))
        i += 1
    logging.info(f'Merged POS tags: {merged_tags}')
    return merged_tags

def extract_keywords(text, n_keywords=5):
    logging.info(f'Extracting {n_keywords} keywords')
    logging.info(f'Input text: {text}')

    try:
        pos_tags = pos_tagging(text)
        stop_words = [word.lower() for word, tag in pos_tags
                      if tag not in ['NOUN', 'PROPN', 'ADJ', 'VERB', 'ADV']]
        logging.info(f'Stopwords: {stop_words}')

        keywords = kw_model.extract_keywords(text, top_n=n_keywords,
                                             stop_words=stop_words)
        logging.info(f'Keywords: {keywords}')
        return '\n'.join([str(kw) for kw, _ in keywords])
    except Exception as e:
        logging.error(f'Error: {e}')
        return 'Error: Could not extract keywords'

def create_interface():
    iface = gr.Interface(
            fn=extract_keywords,
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

