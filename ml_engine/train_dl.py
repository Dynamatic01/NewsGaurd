import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import pickle

DATA_FILE = "fake_or_real_news.csv"
MAX_VOCAB_SIZE = 20000
MAX_SEQ_LENGTH = 500

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

def plot_history(history, filename):
    plt.figure(figsize=(10,4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved training history plot to {filename}")

def main():
    if not os.path.exists(DATA_FILE):
        print(f"{DATA_FILE} not found. Please run fetch_data.py first.")
        return

    print("Loading dataset...")
    df = pd.read_csv(DATA_FILE)
    df['label_num'] = df['label'].map({'FAKE': 0, 'REAL': 1})
    df = df.dropna(subset=['label_num', 'text'])
    
    X = df['text'].astype(str)
    y = df['label_num'].astype(int).values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Tokenizing text data...")
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_SEQ_LENGTH, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_SEQ_LENGTH, padding='post', truncating='post')
    
    print("Building LSTM Model...")
    model = Sequential([
        Embedding(input_dim=MAX_VOCAB_SIZE, output_dim=128, input_length=MAX_SEQ_LENGTH),
        LSTM(64, return_sequences=False),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    
    print("\n--- Training LSTM Model ---")
    history = model.fit(
        X_train_pad, y_train,
        epochs=5,
        batch_size=64,
        validation_split=0.1,
        verbose=1
    )
    
    print("\nEvaluating Model...")
    preds_probs = model.predict(X_test_pad)
    preds = (preds_probs > 0.5).astype(int)
    
    acc = accuracy_score(y_test, preds)
    print(f"LSTM Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=['Fake', 'Real']))
    
    cm = confusion_matrix(y_test, preds)
    plot_cm(cm, ['Fake', 'Real'], 'LSTM Confusion Matrix', 'dl_cm.png')
    plot_history(history, 'dl_history.png')
    
    print("\nSaving DL models and tokenizer...")
    model.save('dl_model.keras')
    with open('dl_tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    print("Files saved: dl_model.keras, dl_tokenizer.pkl")

if __name__ == "__main__":
    main()
