import pandas as pd
import re
import string

def load_and_preprocess_data(filepath='spam.csv'):
    """
    Load the SMS Spam Collection dataset and preprocess it.
    """
    df = pd.read_csv(filepath, encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'text']
    df['label_encoded'] = df['label'].map({'ham': 0, 'spam': 1})
    
    print(f"Loaded {len(df)} messages")
    print(f"Spam: {sum(df['label']=='spam')} | Ham: {sum(df['label']=='ham')}")
    return df

def clean_text(text):
    """Basic text cleaning: lowercase + remove punctuation + remove numbers"""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    return ' '.join(text.split())
