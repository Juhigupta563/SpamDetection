from flask import Flask, request, jsonify

import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


app = Flask(__name__)
model = joblib.load('model.pkl')
vectorizer = joblib.load("vectorizer.pkl")


@app.route("/")
def home():
   return render_template('form.html')

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word.isalnum()]
    words = [word for word in words if word not in stopwords.words("english")]
    return " ".join(words)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    message = data.get("message", "")
    cleaned_message = preprocess_text(message)
    transformed_message = vectorizer.transform([cleaned_message])
    prediction = model.predict(transformed_message)[0]
    result = "Spam" if prediction == 1 else "Not Spam"
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)
