from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

def vectorize_text(data, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(data['text'])
    
    # Save the vectorizer to reuse later (e.g., during prediction or deployment)
    os.makedirs("models", exist_ok=True)
    with open("models/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    
    return X
