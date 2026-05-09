import os
from flask import Flask, render_template_string, request, jsonify
import pandas as pd
from dataset import load_and_preprocess_data
from model import train_models, load_model, predict_message
from visualization import create_distribution_plot, create_confusion_matrix_plot
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
import logging
warnings.filterwarnings('ignore')
logging.getLogger('werkzeug').setLevel(logging.ERROR)

app = Flask(__name__)

df = None
model = None
vectorizer = None
X_test = None
y_test = None

HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Spam Classifier</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            background: #f5f7fb;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }
        .card {
            max-width: 700px;
            width: 100%;
            background: white;
            border-radius: 28px;
            box-shadow: 0 12px 30px rgba(0,0,0,0.08);
            overflow: hidden;
            transition: 0.2s;
        }
        .card-header {
            padding: 32px 32px 16px 32px;
            border-bottom: 1px solid #eef2f6;
        }
        .card-header h1 {
            font-size: 1.8rem;
            font-weight: 600;
            color: #1a2c3e;
            letter-spacing: -0.3px;
        }
        .card-body {
            padding: 32px;
        }
        textarea {
            width: 100%;
            border: 1px solid #dce3ec;
            border-radius: 20px;
            padding: 16px 18px;
            font-size: 1rem;
            font-family: inherit;
            resize: vertical;
            background: #ffffff;
            transition: 0.2s;
        }
        textarea:focus {
            outline: none;
            border-color: #3b82f6;
            box-shadow: 0 0 0 3px rgba(59,130,246,0.1);
        }
        button {
            background: #3b82f6;
            color: white;
            border: none;
            padding: 12px 28px;
            font-size: 1rem;
            font-weight: 500;
            border-radius: 40px;
            cursor: pointer;
            margin-top: 20px;
            transition: 0.2s;
            width: auto;
            display: inline-block;
        }
        button:hover {
            background: #2563eb;
            transform: scale(0.98);
        }
        .result-area {
            margin-top: 28px;
            padding: 20px;
            border-radius: 20px;
            background: #f8fafc;
            text-align: center;
            font-size: 1.2rem;
            font-weight: 500;
        }
        .spam {
            background: #fee2e2;
            color: #b91c1c;
            border-left: 5px solid #b91c1c;
        }
        .ham {
            background: #e0f2fe;
            color: #0c4a6e;
            border-left: 5px solid #0c4a6e;
        }
        .confidence-text {
            font-size: 0.85rem;
            opacity: 0.8;
            margin-top: 8px;
        }
        .footer {
            padding: 20px 32px;
            border-top: 1px solid #eef2f6;
            text-align: center;
            font-size: 0.85rem;
            color: #5b6e8c;
            background: #fafcff;
        }
        .footer a {
            color: #3b82f6;
            text-decoration: none;
            margin: 0 6px;
        }
        .footer a:hover {
            text-decoration: underline;
        }
        @media (max-width: 550px) {
            .card-header h1 { font-size: 1.4rem; }
            .card-body { padding: 24px; }
        }
    </style>
</head>
<body>
<div class="card">
    <div class="card-header">
        <h1>📧 Email/SMS Spam Classifier</h1>
    </div>
    <div class="card-body">
        <textarea id="message" rows="5" placeholder="Paste your message here..."></textarea>
        <div style="text-align: center;">
            <button onclick="predict()">Predict</button>
        </div>
        <div id="result" class="result-area" style="display: none;"></div>
    </div>
    <div class="footer">
        Made by Sameer Ahmed &nbsp;|&nbsp;
        <a href="#" id="linkedinLink" target="_blank">LinkedIn</a> &nbsp;|&nbsp;
        <a href="#" id="githubLink" target="_blank">GitHub</a>
    </div>
</div>

<script>
    async function predict() {
        const msg = document.getElementById('message').value.trim();
        if (!msg) {
            alert("Please enter a message.");
            return;
        }
        const resultDiv = document.getElementById('result');
        resultDiv.style.display = 'block';
        resultDiv.innerHTML = 'Analyzing...';
        resultDiv.className = 'result-area';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: msg })
            });
            const data = await response.json();
            const isSpam = data.prediction === 1;
            resultDiv.className = `result-area ${isSpam ? 'spam' : 'ham'}`;
            const label = isSpam ? 'SPAM' : 'HAM';
            resultDiv.innerHTML = `<strong>${label}</strong><br>Confidence: ${data.confidence.toFixed(1)}%<div class="confidence-text">${isSpam ? 'This message looks suspicious.' : 'This appears to be safe.'}</div>`;
        } catch (err) {
            resultDiv.innerHTML = 'Error: could not analyze.';
            resultDiv.style.background = '#ffe6e6';
            console.error(err);
        }
    }

    document.getElementById('linkedinLink').href = "https://www.linkedin.com/in/sameer-ahmed-eng/";
    document.getElementById('githubLink').href = "https://github.com/Sameerr-Ahmed";
</script>
</body>
</html>
"""
@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    msg = data.get('message', '')
    pred, conf = predict_message(model, vectorizer, msg)
    return jsonify({'prediction': int(pred), 'confidence': float(conf)})

def init_app():
    global df, model, vectorizer, X_test, y_test
    print("Loading data...")
    df = load_and_preprocess_data('spam.csv')
    print("Training models...")
    model, vectorizer, X_test, y_test = train_models(df)
    print("Web server starting...")

if __name__ == '__main__':
    init_app()
    print(" Server is running!")
    print("Open this link in your browser:")
    print("http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(debug=False, host='127.0.0.1', port=5000)
