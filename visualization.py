import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import io
import base64

def create_distribution_plot(df):
    """Return base64 image of spam/ham bar chart."""
    counts = df['label'].value_counts()
    fig, ax = plt.subplots(figsize=(8,5))
    colors = ['#2ecc71', '#e74c3c']
    ax.bar(counts.index, counts.values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_title('Message Distribution')
    ax.set_xlabel('Type')
    ax.set_ylabel('Count')
    for i, v in enumerate(counts.values):
        ax.text(i, v+5, str(v), ha='center', fontweight='bold')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=80)
    buf.seek(0)
    img = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    return img

def create_confusion_matrix_plot(model, X_test, y_test):
    """Return base64 image of confusion matrix."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham','Spam'], yticklabels=['Ham','Spam'])
    ax.set_title('Confusion Matrix')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=80)
    buf.seek(0)
    img = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    return img
