from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import joblib
import os

from dataset import clean_text  # cleaning 

def train_models(df):
    """Train Naive Bayes and Logistic Regression, return the best one."""
    print("Cleaning texts...")
    cleaned = [clean_text(msg) for msg in df['text']]
    
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english',
                                 ngram_range=(1,2), max_df=0.8, min_df=2)
    X = vectorizer.fit_transform(cleaned)
    y = df['label_encoded']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    nb = MultinomialNB(alpha=0.5)   #tain
    nb.fit(X_train, y_train)
    
    lr = LogisticRegression(max_iter=1000, random_state=42, 
                            C=1.0, class_weight='balanced')
    lr.fit(X_train, y_train)
    
    nb_f1 = f1_score(y_test, nb.predict(X_test))
    lr_f1 = f1_score(y_test, lr.predict(X_test))
    
    best_model = nb if nb_f1 >= lr_f1 else lr
    best_name = "Naive Bayes" if nb_f1 >= lr_f1 else "Logistic Regression"
    print(f"Best model: {best_name} (F1={max(nb_f1, lr_f1):.4f})")
    
    os.makedirs('models', exist_ok=True) #save 
    joblib.dump(best_model, 'models/spam_model.pkl')
    joblib.dump(vectorizer, 'models/vectorizer.pkl')
    
    return best_model, vectorizer, X_test, y_test

def load_model():
    """Load saved model and vectorizer if they exist."""
    try:
        model = joblib.load('models/spam_model.pkl')
        vec = joblib.load('models/vectorizer.pkl')
        return model, vec
    except:
        return None, None

def predict_message(model, vectorizer, message):
    """Return prediction (0=ham,1=spam) and confidence percentage."""
    cleaned = clean_text(message)
    vec_msg = vectorizer.transform([cleaned])
    pred = model.predict(vec_msg)[0]
    prob = model.predict_proba(vec_msg)[0]
    confidence = prob[pred] * 100
    return pred, confidence
