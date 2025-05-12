from flask import Flask, render_template, request
import joblib
from src.preprocess import preprocess_text
from src.predict import predict_news

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['news']
    processed_text = preprocess_text(text)
    result = predict_news(processed_text)

    if result is None:
        return render_template("index.html", prediction="Error: could not classify the news.")

    label, confidence = result
    return render_template('index.html', prediction=label, confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)
