import pandas                        as pd
import unidecode
import pickle
from sklearn.linear_model            import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection         import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from string                          import punctuation
from flask                           import Flask, request, render_template

app = Flask(__name__)

vetorizar = CountVectorizer(lowercase=False, ngram_range = (1,2))

tfidf = TfidfVectorizer(lowercase=False, max_features=50)

# token_espaco    = tokenize.WhitespaceTokenizer()
# token_pontuacao = tokenize.WordPunctTokenizer()

# data = pd.read_csv("../[TRATADO]imdb-reviews.csv")

pontuacao = list()
for ponto in punctuation:
        pontuacao.append(ponto)

# palavras_irrelevantes_sem_acentos = [unidecode.unidecode(texto) for texto in palavras_irrelevantes]
# stopWords_pontuacao_acentos = pontuacao + palavras_irrelevantes + [unidecode.unidecode(texto) for texto in palavras_irrelevantes];

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/treinar')
def treinarModelo():
    # data = pd.read_csv("../[TRATADO]imdb-reviews.csv")

    data = 'Troque quando necessário...'

    print('VETORIZANDO')

    vetorizar = CountVectorizer(lowercase=False, ngram_range = (1,2))
    palavras = vetorizar.fit_transform(data["tratamento_5"])

    treino, teste, class_treino, class_teste = train_test_split(palavras,data["classificacao"],random_state = 42)
                                                
    print('TREINANDO')

    model = LogisticRegression(max_iter=1000).fit(treino, class_treino)


    # Save the vectorizer
    vec_file = 'vectorizer.pickle'
    pickle.dump(vetorizar, open(vec_file, 'wb'))

    # Save the model
    mod_file = 'classification.model'
    pickle.dump(model, open(mod_file, 'wb'))

    print('MODELO SALVO!')

    result = 'MODELO SALVO | SCORE: ',  model.score(teste, class_teste)

    return result

#Load the classification model from disk and use for predictions
def classify_utterance(utt):
    # load the vectorizer
    loaded_vectorizer = pickle.load(open('/var/www/html/NLTK_IA_API/src/vectorizer.pickle', 'rb'))

    # load the model
    loaded_model = pickle.load(open('/var/www/html/NLTK_IA_API/src/classification.model', 'rb'))

    # make a prediction

    return loaded_model.predict(loaded_vectorizer.transform(utt))

@app.route('/validar')
def validar():
    frasesInUrl = request.args.getlist("validar")
    dicionario = {'frases': frasesInUrl}
    df = pd.DataFrame(dicionario, columns=['frases'])
    predict = classify_utterance(df["frases"])
    retorno = str(predict)
    return retorno
