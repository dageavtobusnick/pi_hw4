from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import streamlit as st


def load_model():
    model_name = "Helsinki-NLP/opus-mt-en-ru"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    tokenizer.save_pretrained(Path.cwd() / 'model' / 'en_ru_local')
    model.save_pretrained(Path.cwd() / 'model' / 'en_ru_local')
    
if not Path(Path.cwd() / 'model').exists():
    load_model()

tokenizer = AutoTokenizer.from_pretrained(Path.cwd() / 'model' / 'en_ru_local')
model = AutoModelForSeq2SeqLM.from_pretrained(Path.cwd() / 'model' / 'en_ru_local')

def translate_phrase(phrase):
    inputs = tokenizer(phrase, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=100)
    out_text = tokenizer.batch_decode(output, skip_special_tokens=True)
    return out_text[0]

prompt = st.chat_input("Введите фразу для перевода:")
if prompt:
    st.write(translate_phrase(prompt))