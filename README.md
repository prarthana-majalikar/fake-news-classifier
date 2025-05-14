# 📰 Fake News Classifier 🔍

A web-based application that uses machine learning and NLP techniques to detect whether a given news article is **real** or **fake**.

![Fake News Classifier Demo]
<img width="1065" alt="image" src="https://github.com/user-attachments/assets/9fe5169d-10cc-43c8-98e9-1bc5c002214b" />


---

## 🚀 Features

- 🔤 **Text input** for news content.
- 🧠 **ML model** trained on real-world fake vs. real news datasets.
- 📊 **Confidence score** showing prediction certainty.
- 🌐 **Web interface** using Flask.
- 🔄 Clean and modular codebase (preprocessing, prediction, UI).

---

## 🧩 Tech Stack

- **Frontend**: HTML, CSS
- **Backend**: Python (Flask)
- **ML/NLP**: scikit-learn, NLTK, TfidfVectorizer
- **Model**: Logistic Regression (can be swapped with others)

---

---

## ⚙️ Installation & Setup

1. **Clone the repo**

   ```bash
   git clone https://github.com/your-username/fake-news-classifier.git
   cd fake-news-classifier
   
2. Install dependencies
   ```bash
   pip install -r requirements.txt

3. Run the app
      ```bash
      PYTHONPATH=. python app/app.py

4. Open your browser and go to http://127.0.0.1:5000


---
🔬 How It Works
1. User submits a news article snippet/headline.
2. The input is preprocessed (tokenized, cleaned, stemmed).
3. The ML model classifies it as FAKE or REAL.
4. The result and confidence score are displayed on the UI.


---
🛠️ Troubleshooting
- ModuleNotFoundError: Make sure you’re using the root directory as PYTHONPATH or properly set relative imports.
- Model not loading: Check that model.pkl exists in the /models directory.

---

✨ Future Improvements
- Replace Logistic Regression with a deep learning model (LSTM/BERT).
- Add user authentication & fake news reporting.
- Deploy to Heroku or Render with Docker.

---

🙋‍♀️ Author
- Made by Prarthana Majalikar

