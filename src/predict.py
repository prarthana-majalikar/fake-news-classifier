import joblib
from src.preprocess import preprocess_text

# Load model and vectorizer
model = joblib.load('models/fake_news_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

def predict_news(text):
    # Preprocess the input text
    processed = preprocess_text(text)
    print(f"\nPreprocessed headline: {processed}")
    
    # Vectorize
    vectorized = vectorizer.transform([processed])
    
    # Predict
    prediction = model.predict(vectorized)[0]
    prob = model.predict_proba(vectorized).max()
    
    # Output
    label = "REAL" if prediction == 1 else "FAKE"
    confidence = round(prob* 100, 2)
    print(f"\nPrediction: {label} news ({confidence}% confidence)")
    return label, confidence

if __name__ == "__main__":
    print("Enter a news headline or article to classify it as FAKE or REAL.")
    while True:
        user_input = input("\nEnter text (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        predict_news(user_input)
