from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import pandas as pd
from preprocess import preprocess_data
from data_loader import load_data
from vectorizer import vectorize_text
import joblib

def train_model():
    # Load and preprocess data
    data = load_data('data/Fake.csv', 'data/True.csv')
    data = preprocess_data(data)
    
    # Vectorize text
    X = vectorize_text(data)
    y = data['label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Save model
    with open("models/fake_news_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save model and vectorizer
    joblib.dump(model, 'models/fake_news_model.pkl')
    joblib.dump(vectorizer, 'models/vectorizer.pkl')

if __name__ == "__main__":
    train_model()
