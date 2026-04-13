import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
import pickle
import os
import argparse
import time

MAX_WORDS = 20000
MAX_SEQ_LENGTH = 500
EMBEDDING_DIM = 64

def train_custom_data(csv_path: str, text_col: str, label_col: str, epochs: int):
    """
    Trains the LSTM Neural Network on a custom CSV entirely automatically.
    The labels should map so that Fake News represents 0, and Real News represents 1.
    """
    if not os.path.exists(csv_path):
        print(f"Error: Could not find CSV at {csv_path}")
        return

    print(f"\n--- Loading Custom Data: {csv_path} ---")
    df = pd.read_csv(csv_path)
    
    if text_col not in df.columns or label_col not in df.columns:
        print(f"Error: Your CSV must contain columns '{text_col}' and '{label_col}'.")
        print(f"Found columns: {df.columns.tolist()}")
        return
        
    df.dropna(subset=[text_col, label_col], inplace=True)
    X_raw = df[text_col].astype(str).tolist()
    y_raw = pd.to_numeric(df[label_col], errors='coerce').fillna(0).astype(int).tolist()
    
    print(f"Loaded {len(X_raw)} rows of training data.")
    
    # 1. Tokenization Setup
    print("Building vocabulary embeddings...")
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_raw)
    
    # Save Tokenizer immediately
    with open('dl_tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
        
    print("Vocabulary computed and Tokenizer Saved -> dl_tokenizer.pkl")
    
    # 2. Sequence Padding
    X_seq = tokenizer.texts_to_sequences(X_raw)
    X_pad = pad_sequences(X_seq, maxlen=MAX_SEQ_LENGTH, padding='post', truncating='post')
    y = np.array(y_raw)
    
    # 3. Model Architecture Construction
    print("Constructing Deep Neural Network Architecture...")
    model = Sequential([
        Embedding(input_dim=MAX_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_SEQ_LENGTH),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    
    # 4. Training
    print(f"Commencing Training for {epochs} epochs...")
    start_time = time.time()
    
    history = model.fit(
        X_pad, y,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2, # Automatically keeps 20% back to ensure it doesn't overfit
        verbose=1
    )
    
    end_time = time.time()
    print(f"Training Complete in {round(end_time - start_time, 2)} seconds!")
    
    # 5. Export
    model.save('dl_model.keras')
    print("New Neural Network weights exported -> dl_model.keras")
    print("\n--- The API is now ready to use your custom trained model! ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bring Your Own Data - LSTM Trainer")
    parser.add_argument("--csv", type=str, default="custom_data.csv", help="Path to your CSV file")
    parser.add_argument("--text-col", type=str, default="text", help="Name of the text column")
    parser.add_argument("--label-col", type=str, default="label", help="Name of the label column (0=Fake, 1=Real)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of times to process the dataset")
    
    args = parser.parse_args()
    train_custom_data(args.csv, args.text_col, args.label_col, args.epochs)
