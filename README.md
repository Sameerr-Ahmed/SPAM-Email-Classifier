# 📧 Email/SMS Spam Classifier

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-%23000.svg?style=flat&logo=flask&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)

A high-performance machine learning application designed to detect and classify spam messages in emails and SMS. Built with Python, this project leverages Natural Language Processing (NLP) techniques, specifically TF-IDF vectorization, combined with powerful classification algorithms to accurately filter unwanted communications.

---

## ✨ Key Features

* **Dual Interface:** Accessible via a user-friendly web interface (Flask) or a lightweight terminal-based CLI.
* **Algorithm Comparison:** Utilizes both Naive Bayes and Logistic Regression algorithms to ensure high accuracy.
* **Real-Time Predictions:** Enter text and receive immediate SPAM or HAM (legitimate) classifications along with confidence scores.
* **Automated Visualizations:** Built-in tools to visualize dataset distributions and model statistics.

---

## 📊 Model Performance

Our model has been trained and validated on a robust dataset, yielding exceptional metrics:

| Metric | Score |
| :--- | :--- |
| **Training Accuracy** | 98.5% |
| **F1-Score** | 94.2% |
| **Best Performing Model** | Logistic Regression |

**Dataset Overview:**
* **Total Messages:** 5,572
* **Ham (Legitimate):** 4,825
* **Spam:** 747

---

## 📂 Project Structure

```text
├── app.py               # Main Flask web application interface
├── main.py              # Terminal-based (CLI) alternative version
├── model.py             # Core ML logic (training and predictions)
├── dataset.py           # Data ingestion, cleaning, and preprocessing
├── visualization.py     # Generation of spam/ham distribution charts
├── spam.csv             # Raw dataset file
└── models/              # Directory for serialized/trained ML models