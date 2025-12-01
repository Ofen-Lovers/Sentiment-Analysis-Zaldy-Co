import nbformat as nbf

nb = nbf.v4.new_notebook()

# Cell 1: Title
text_1 = """# Week 14 - Lab Sentiment Analysis
## Goal
Perform sentiment analysis on Facebook comments using Lexicon-based (VADER, TextBlob) and Machine Learning-based (TF-IDF + Logistic Regression/Naive Bayes) approaches.
"""

# Cell 2: Imports
code_2 = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
import emoji
from deep_translator import GoogleTranslator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Setup
tqdm.pandas()
sns.set_style("whitegrid")
"""

# Cell 3: Load Data
code_3 = """# Load Data
try:
    df = pd.read_csv('data/raw.csv')
    print(f"Loaded {len(df)} comments.")
except Exception as e:
    print(f"Error loading data: {e}")
    # Fallback if file not found or error
    df = pd.DataFrame(columns=['comments', 'date'])

df.head()
"""

# Cell 4: Preprocessing
code_4 = """# Preprocessing

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user mentions (e.g. @username) - though FB comments might not have standard @
    text = re.sub(r'@\w+', '', text)
    return text

def preprocess_comment(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Demojize (convert emojis to text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    
    # 2. Basic cleaning
    text = clean_text(text)
    
    # 3. Lowercasing
    text = text.lower()
    
    # 4. Tokenization (simple split or nltk)
    # We will use simple split for now to keep it fast, or nltk word_tokenize
    # tokens = nltk.word_tokenize(text) # Can be slow for many comments
    
    # 5. Remove punctuation (optional, but VADER handles punctuation well, so maybe keep it for VADER?)
    # For ML, we usually remove it. Let's keep a "clean_for_ml" version and "raw_for_vader" version.
    # But instructions say: Cleaning (punctuation, URLs), Tokenization, Stopword removal, Lemmatization.
    
    # Let's do standard cleaning for the 'cleaned_text' column
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    
    tokens = text.split()
    
    # 6. Stopword removal
    stop_words = set(stopwords.words('english')) # Note: This assumes English text. We translate first?
    # Strategy: Translate FIRST, then clean.
    
    return text

# Translation Wrapper
def translate_comment(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    try:
        # Simple heuristic: if text is very short or looks like English, skip?
        # But mixed text is hard. Let's try to translate everything to English.
        # deep_translator limits: 5000 chars.
        if len(text) > 4500:
            text = text[:4500]
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception as e:
        return text # Return original if failed

"""

# Cell 5: Apply Preprocessing
code_5 = """# Apply Preprocessing
import os

if os.path.exists('outputs/data/cleaned_comments.csv'):
    print("Found existing cleaned_comments.csv. Loading...")
    df = pd.read_csv('outputs/data/cleaned_comments.csv')
    # Ensure columns exist
    if 'translated_comments' not in df.columns:
        print("Existing file missing 'translated_comments'. Re-processing...")
        run_processing = True
    else:
        run_processing = False
else:
    run_processing = True

if run_processing:
    # 1. Deduplication
    print(f"Original count: {len(df)}")
    df.drop_duplicates(subset=['comments'], inplace=True)
    print(f"Count after deduplication: {len(df)}")

    # 2. Translation (This takes time!)
    print("Translating comments... (this may take a while)")
    df['translated_comments'] = df['comments'].progress_apply(translate_comment)

    # 3. Cleaning & Lemmatization
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def process_final(text):
        if not isinstance(text, str): return ""
        # Demojize first to keep emoji meanings if they were translated or kept
        text = emoji.demojize(text, delimiters=(" ", " "))
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        text = text.lower()
        tokens = text.split()
        # Stopwords & Lemmatization
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        return " ".join(tokens)

    df['cleaned_comments'] = df['translated_comments'].progress_apply(process_final)

    # Save cleaned data
    df.to_csv('outputs/data/cleaned_comments.csv', index=False)
    print("Saved cleaned_comments.csv")

df.head()
"""

# Cell 6: Lexicon Analysis
code_6 = """# Lexicon-Based Sentiment Analysis

analyzer = SentimentIntensityAnalyzer()

def get_vader_score(text):
    if not isinstance(text, str) or not text.strip():
        return 'Neutral'
    # VADER works best on raw text (with punctuation/caps), but we have translated text.
    # We will use the translated text (before full cleaning) if possible, or just cleaned.
    # Let's use 'translated_comments' which has punctuation.
    score = analyzer.polarity_scores(text)
    compound = score['compound']
    if compound >= 0.05:
        return 'Positive'
    elif compound <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def get_textblob_score(text):
    if not isinstance(text, str) or not text.strip():
        return 'Neutral'
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

df['vader_sentiment'] = df['translated_comments'].apply(get_vader_score)
df['textblob_sentiment'] = df['translated_comments'].apply(get_textblob_score)

# Save results
df.to_csv('outputs/data/lexicon_results.csv', index=False)

# Visualization
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.countplot(x='vader_sentiment', data=df, order=['Positive', 'Neutral', 'Negative'])
plt.title('VADER Sentiment Distribution')

plt.subplot(1, 2, 2)
sns.countplot(x='textblob_sentiment', data=df, order=['Positive', 'Neutral', 'Negative'])
plt.title('TextBlob Sentiment Distribution')

plt.tight_layout()
plt.savefig('outputs/images/sentiment_distribution.png')
plt.show()
"""

# Cell 7: ML Analysis
code_7 = """# ML-Based Sentiment Analysis
# We use VADER and TextBlob labels as Ground Truth targets to compare how well ML models can learn them.

print("Handling NaN values...")
# Ensure no NaNs and all are strings
df['cleaned_comments'] = df['cleaned_comments'].fillna('')
df['cleaned_comments'] = df['cleaned_comments'].astype(str)

targets = ['vader_sentiment', 'textblob_sentiment']
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB()
}

X = df['cleaned_comments']
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(X).toarray()

results_txt = ""

for target in targets:
    print(f"\\n{'='*40}\\nTarget: {target}\\n{'='*40}")
    results_txt += f"\\n{'='*40}\\nTarget: {target}\\n{'='*40}\\n"
    
    y = df[target]
    
    # Filter out NaN if any in target
    valid_idx = y.notna()
    X_subset = X_tfidf[valid_idx]
    y_subset = y[valid_idx]

    X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.2, random_state=42)

    for name, model in models.items():
        print(f"Training {name} on {target}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        print(f"--- {name} ---")
        print(f"Accuracy: {acc:.4f}")
        print(report)
        
        results_txt += f"--- {name} ---\\n"
        results_txt += f"Accuracy: {acc:.4f}\\n"
        results_txt += f"Classification Report:\\n{report}\\n\\n"
        
        # Confusion Matrix
        unique_labels = sorted(y_subset.unique())
        cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
        plt.title(f'Confusion Matrix - {name} ({target})')
        plt.ylabel(f'True Label ({target})')
        plt.xlabel('Predicted Label')
        plt.savefig(f'outputs/images/confusion_matrix_{name.replace(" ", "_").lower()}_{target}.png')
        plt.show()

with open('outputs/data/ml_results.txt', 'w') as f:
    f.write(results_txt)
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_1),
    nbf.v4.new_code_cell(code_2),
    nbf.v4.new_code_cell(code_3),
    nbf.v4.new_code_cell(code_4),
    nbf.v4.new_code_cell(code_5),
    nbf.v4.new_code_cell(code_6),
    nbf.v4.new_code_cell(code_7)
]

nbf.write(nb, 'sentiment_analysis.ipynb')
print("Notebook created successfully.")
