from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from flask import Flask, request, render_template
from string import punctuation

import pandas as pd
import pickle

app = Flask(__name__)

vetorizar = CountVectorizer(lowercase=False, ngram_range=(1, 2))

tfidf = TfidfVectorizer(lowercase=False, max_features=50)

pontuacao = list()
for ponto in punctuation:
    pontuacao.append(ponto)

@app.route('/')
def home():
    return render_template("index.html")

# Load the classification model and use for predictions
def classify_utterance(utt):
    # load the vectorizer
    loaded_vectorizer = pickle.load(open('./vectorizer.pickle', 'rb'))

    # load the model
    loaded_model = pickle.load(open('./classification.model', 'rb'))

    # make a prediction
    return loaded_model.predict(loaded_vectorizer.transform(utt))


@app.route('/validar')
def validar():
    df = pd.DataFrame({'frases': request.args.getlist("validar")}, columns=['frases'])
    return str(classify_utterance(df["frases"]))
