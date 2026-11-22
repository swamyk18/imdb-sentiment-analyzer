import re
from pathlib import Path
import pandas as pd 
import numpy as np
import streamlit as st
from gensim.models import Word2Vec
import tensorflow as tf
from tensorflow.keras.models import load_model

VECTOR_SIZE=100
MODEL_DIR = Path(__file__).parent / "models"

st.set_page_config(page_title='Movie Review Sentiment',page_icon='🎬')


@st.cache_resource
def load_models():
    w2v = Word2Vec.load(str(MODEL_DIR / "word2vec.model"))      # gensim loader
    ann = load_model(str(MODEL_DIR / "sentiment_ann.keras"))    # Keras loader
    return w2v, ann

def data_cleaning(text):
    text = text.lower()
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"n't", " not", text)      # wouldn't → would not, didn't → did not
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'m", " am", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def avg_vectors(tokens,model,vector_size=VECTOR_SIZE):
    vector=[model.wv[word] for word in tokens if word in model.wv]
    avg_vectors=np.mean(vector,axis=0) if vector else np.zeros(vector_size)
    return avg_vectors



st.title('🎬 Movie Review Sentiment Analyzer')


try:
    word2vec_model,ann_model=load_models()
except Exception as e:
    st.error('Could not load the model files.')
    st.exception(e)
    st.stop()

review=st.text_area('Enter a movie review ',height=150,placeholder="This movie was fantastic, I loved every minute...")

if st.button('predict Sentiment',type='primary'):
    if not review.strip():
        st.warning('Please type as review first.')
    else:
        tokens=data_cleaning(review).split()
        known = [w for w in tokens if w in word2vec_model.wv]
        if not known:
            st.warning("None of the words were recognised by the model, "
                       "so the result is unreliable. Try a longer review.")

        vec=avg_vectors(tokens,word2vec_model).reshape(1,-1)
        prob=float(ann_model.predict(vec,verbose=0,)[0][0])

        if prob>0.5:
            st.success(f"😀 Positive  (confidence {prob:.1%})")  
        else:
              st.error(f"😞 Negative  (confidence {1 - prob:.1%})")
               
        st.progress(prob, text=f"Probability of positive: {prob:.3f}")

    


