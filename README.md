# SPAM-Email-Classifier
Email/SMS Spam Classifier - A Machine Learning web app that detects spam messages with 98% accuracy. Users type any email or SMS and instantly get SPAM/HAM result with confidence score. Built with Python, Flask, and scikit-learn.

## Project Overview
This project detects spam messages in emails and SMS using Machine Learning. It uses TF-IDF vectorization with two powerful algorithms (Naive Bayes & Logistic Regression) to classify messages as SPAM or HAM with high accuracy.

## Model Performance
__ Training Accuracy: 98.5%
__ Best Model: Logistic Regression
__ F1-Score: 94.2%
__ Dataset: 5,572 messages (747 spam, 4,825 ham)

## Installation Commands

Open terminal in the project folder and run these commands one by one:

pip install flask
pip install pandas
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install joblib

Or install all together with this single command:

pip install flask pandas scikit-learn matplotlib seaborn joblib

## How to Run This Project

### 1. Run the Web Application
python app.py

### 2. Alternative: Terminal Version (No GUI)
python main.py

### 3. View Dataset Statistics
The web app at http://localhost:5000 shows all statistics automatically

## Files in This Project
__ app.py - Main web interface 
__ dataset.py - Loads and cleans the spam dataset
__ model.py - Trains ML models and makes predictions
__ visualization.py - Shows spam/ham distribution charts
__ main.py - Terminal-based version 
__ spam.csv - Dataset file 
__ models/ - Folder that saves trained model

## How to Test the Spam Detector
1. Run: python app.py
2. Open browser to: http://localhost:5000
3. Type any message in the text box
4. Click "Predict" button
5. See result: SPAM  or HAM  with confidence score

## Example Messages to Test

### Spam Examples (Will show SPAM):
- "CONGRATULATIONS! You won $1,000,000! Click here to claim"
- "URGENT: Your account will be suspended. Verify now"
- "FREE iPhone! Limited time offer. Send this to 10 friends"

### Ham Examples (Will show HAM):
- "Hey, want to grab coffee tomorrow at 3pm?"
- "The meeting has been moved to 2pm. Please update"
- "Thanks for your purchase. Your order will arrive Friday"

## Troubleshooting Commands

If you get "No module named flask" error:
pip install flask

If you need to upgrade pip:
python -m pip install --upgrade pip

If you want to check Python version:
python --version

If port 5000 is already in use:
Change port in app.py (last line) from 5000 to 5001

If you want to stop the server:
Press Ctrl + C in terminal

## Author
**Sameer Ahmed**

## Project Completed
April 2026
