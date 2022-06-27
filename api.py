from flask import Flask, request, jsonify
import nltk
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import joblib

# DOWNLOADING NLTK PACKAGES
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
app = Flask(__name__)

model = joblib.load("data/model.pkl")
Tfidf_vect = joblib.load("data/vectorizer.pkl")

@app.route('/predict/detect-fear', methods=['GET'])
def respond():
    # Retrieve the name from the url parameter /getmsg/?name=
    inp = request.args.get("text", None)
    
    status_code = 500
    res = ""
    try:
        inp_vec = Tfidf_vect.transform([inp])
        pred = model.predict(inp_vec)
        if pred == 0:
            res = "Normal"
        else:
            res = "Fear speech"
        status_code = 200
    except Exception as e:
        res = "Error: " + e

    response = {
        "status": status_code,
        "result": res,
    }


    # Return the response in json format
    return jsonify(response)

@app.route('/')
def index():
    # A welcome message to test our server
    return "<h1>Welcome to the API endpoint</h1>"


if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)