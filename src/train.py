from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score, confusion_matrix, roc_curve, roc_auc_score
import pickle
import pandas as pd
from preprocess import preprocess_data
from data_loader import load_data
from vectorizer import vectorize_text
import joblib
import matplotlib.pyplot as plt

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

    y_probability = model.predict_proba(X_test)[:, 1]
    # Evaluation Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probability)
    print("Accuracy:", round(acc, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC-AUC Score:", round(roc_auc, 4))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save evaluation metrics
    with open("metrics_report.txt", "w") as f:
        f.write(f"Accuracy: {acc}\nF1 Score: {f1}\nROC-AUC: {roc_auc}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_test, y_pred))

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_probability)
    # Compute AUC
    roc_auc = roc_auc_score(y_test, y_probability)

    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Save model and vectorizer
    joblib.dump(model, 'models/fake_news_model.pkl')

if __name__ == "__main__":
    train_model()
