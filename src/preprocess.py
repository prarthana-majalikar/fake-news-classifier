import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize text
    words = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Lemmatize words
    # lemmatizer = WordNetLemmatizer()
    # words = [lemmatizer.lemmatize(word) for word in words]
    
    return " ".join(words)


def preprocess_data(df):
    df['text'] = df['text'].apply(preprocess_text)
    return df

# testing
if __name__ == "__main__":
    from data_loader import load_data
    fake_path = 'data/Fake.csv'
    true_path = 'data/True.csv'
    original_data = load_data(fake_path, true_path)
    preprocessed_data = preprocess_data(original_data.copy())
    for i in range(5):
        print(f"\nOriginal:    {original_data.iloc[i]['text']}")
        print(f"Preprocessed:{preprocessed_data.iloc[i]['text']}")
