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
    text = request.form['headline']
    processed_text = preprocess_text(text)
    result = predict_news(processed_text)
    if result:
        label, confidence = result
        return render_template('index.html', label=label, confidence=confidence, user_input=text)
    else:
        return render_template('index.html', label="Error", confidence="0.0", user_input=text)

if __name__ == "__main__":
    app.run(debug=True)
