import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import os

DATA_FILE = "fake_or_real_news.csv"

def plot_cm(cm, classes, title, filename):
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved confusion matrix plot to {filename}")

def main():
    if not os.path.exists(DATA_FILE):
        print(f"{DATA_FILE} not found. Please run fetch_data.py first.")
        return

    print("Loading dataset...")
    df = pd.read_csv(DATA_FILE)
    
    # Dataset structure: id, title, text, label 
    # Labels are typically "FAKE" and "REAL"
    print("Mapping labels: FAKE -> 0, REAL -> 1")
    df['label_num'] = df['label'].map({'FAKE': 0, 'REAL': 1})
    df = df.dropna(subset=['label_num', 'text']) # Drop invalid rows
    
    X = df['text']
    y = df['label_num'].astype(int)
    
    print("Splitting dataset 80% train / 20% test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("TF-IDF Vectorization...")
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, max_features=10000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # 1. Logistic Regression
    print("\n--- Training Logistic Regression ---")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_tfidf, y_train)
    lr_pred = lr_model.predict(X_test_tfidf)
    lr_acc = accuracy_score(y_test, lr_pred)
    print(f"Logistic Regression Accuracy: {lr_acc:.4f}")
    
    # 2. Random Forest
    print("\n--- Training Random Forest ---")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_tfidf, y_train)
    rf_pred = rf_model.predict(X_test_tfidf)
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f"Random Forest Accuracy: {rf_acc:.4f}")
    
    # Evaluate best
    best_model = lr_model if lr_acc >= rf_acc else rf_model
    best_name = "Logistic Regression" if lr_acc >= rf_acc else "Random Forest"
    best_pred = lr_pred if lr_acc >= rf_acc else rf_pred
    best_acc = max(lr_acc, rf_acc)
    
    print(f"\nBest Model: {best_name} (Acc: {best_acc:.4f})")
    print(classification_report(y_test, best_pred, target_names=['Fake', 'Real']))
    
    cm = confusion_matrix(y_test, best_pred)
    plot_cm(cm, ['Fake', 'Real'], f'{best_name} Confusion Matrix', 'classical_cm.png')
    
    print("\nSaving models to .pkl files...")
    joblib.dump(best_model, 'classical_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    print("Files saved: classical_model.pkl, tfidf_vectorizer.pkl")

if __name__ == "__main__":
    main()
