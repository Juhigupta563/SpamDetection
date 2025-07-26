from flask import Flask, request, render_template
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import os

# Set NLTK data path
nltk_data_path = os.path.join(os.path.dirname(__file__), "nltk_data")
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)

# Initialize Flask app
app = Flask(__name__)
try:
    model = joblib.load("spam_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except FileNotFoundError:
    print("Error: spam_model.pkl or vectorizer.pkl not found in project directory.")
    exit(1)

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word.isalnum()]
    words = [word for word in words if word not in stopwords.words("english")]
    return " ".join(words)

@app.route("/")
def home():
    return render_template("form.html")

@app.route("/predict", methods=["POST"])
def predict():
    message = request.form.get("message", "")
    try:
        cleaned_message = preprocess_text(message)
        transformed_message = vectorizer.transform([cleaned_message])
        prediction = model.predict(transformed_message)[0]
        result = "SPAM 🚫" if prediction == 1 else "Not Spam ✅"
    except Exception as e:
        result = f"Error processing message: {str(e)}"
    return render_template("form.html", prediction=result, message=message)

if __name__ == "__main__":
    app.run(debug=True)
