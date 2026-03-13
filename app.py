import numpy as np
import re
import pickle
from flask import Flask, render_template, request
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

nltk.download('stopwords')

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open('finalized_model.sav', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.sav', 'rb'))

port_stem = PorterStemmer()


def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()

    stemmed_content = [
        port_stem.stem(word)
        for word in stemmed_content
        if not word in stopwords.words('english')
    ]

    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    news = request.form['news']

    news = stemming(news)

    vector = vectorizer.transform([news])

    prediction = model.predict(vector)

    if prediction[0] == 0:
        result = "Real News ✅"
    else:
        result = "Fake News ❌"

    return render_template('index.html', prediction_text=result)


if __name__ == "__main__":
    app.run(debug=True)