import random

import editdistance
import full_pun_generation.pronunciation as pron
import gradio as gr


def translate(text):
    grapheme_words = text.split(" ")
    pronunciation = pron.get_pronunciation(text)
    phonetic_words = pronunciation.split(" ")

    new_sentence = []
    for i, word in enumerate(phonetic_words):
        all_writings, _ = pron.phoneme_to_grapheme(word)
        weights = [editdistance.eval(grapheme_words[i], w) for w in all_writings]
        weights = [1 / (w + 1) for w in weights]

        new_word = ""
        new_word_pron = ""
        distance = 10
        while not new_word or distance > 4:
            new_word = random.choices(all_writings, weights=weights, k=1)[0]
            new_word_pron = pron.get_pronunciation(new_word)
            distance = editdistance.eval(new_word_pron, word)
            print(grapheme_words[i], new_word, word, new_word_pron, distance)
        new_sentence.append(new_word)
    new_sentence = " ".join(new_sentence)
    return new_sentence.strip()


iface = gr.Interface(
    fn=translate, inputs=gr.Textbox(), outputs=gr.Textbox(show_copy_button=True)
)
iface.launch()
