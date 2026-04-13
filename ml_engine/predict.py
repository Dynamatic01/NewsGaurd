import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

# Thresholds for Suspicious
LOWER_BOUND = 0.40
UPPER_BOUND = 0.60
MAX_SEQ_LENGTH = 500

class FakeNewsPredictor:
    def __init__(self, mode='classical'):
        self.mode = mode
        self.model = None
        self.vectorizer = None
        self.tokenizer = None
        
        if mode == 'classical':
            self._load_classical()
        elif mode == 'dl':
            self._load_dl()
        else:
            raise ValueError("Mode must be 'classical' or 'dl'")

    def _load_classical(self):
        try:
            self.model = joblib.load('classical_model.pkl')
            self.vectorizer = joblib.load('tfidf_vectorizer.pkl')
            print("Loaded Classical ML model and vectorizer.")
        except Exception as e:
            print("Could not load classical models. Ensure train_classical.py has been run.")
            print(f"Error: {e}")

    def _load_dl(self):
        try:
            self.model = tf.keras.models.load_model('dl_model.keras')
            with open('dl_tokenizer.pkl', 'rb') as f:
                self.tokenizer = pickle.load(f)
            print("Loaded Deep Learning LSTM model and tokenizer.")
        except Exception as e:
            print("Could not load DL models. Ensure train_dl.py has been run.")
            print(f"Error: {e}")

    def predict(self, text):
        if not text or not isinstance(text, str):
            return "Invalid text.", 0.0
            
        if self.mode == 'classical':
            # TF-IDF
            X_vec = self.vectorizer.transform([text])
            # Getting probability of class 1 (REAL)
            prob = self.model.predict_proba(X_vec)[0][1]
        else:
            # DL sequence padding
            seq = self.tokenizer.texts_to_sequences([text])
            X_pad = pad_sequences(seq, maxlen=MAX_SEQ_LENGTH, padding='post', truncating='post')
            prob = self.model.predict(X_pad, verbose=0)[0][0]
            
        # Classification Logic
        if prob > UPPER_BOUND:
            label = "Real"
        elif prob < LOWER_BOUND:
            label = "Fake"
        else:
            label = "Suspicious"
            
        return label, float(prob)


if __name__ == "__main__":
    test_texts = [
        "Breaking: Alien spaceships have officially landed in New York City according to anonymous sources on Reddit.",
        "The Federal Reserve announced on Wednesday that it will be keeping interest rates steady for the next quarter.",
        "Some politicians are suggesting that the new tax plan might have unintended consequences for local businesses, though specific details are still being debated."
    ]

    print("--- Using Classical Pipeline ---")
    predictor_cls = FakeNewsPredictor(mode='classical')
    if predictor_cls.model:
        for t in test_texts:
            label, score = predictor_cls.predict(t)
            print(f"Text Snippet: '{t[:60]}...'")
            print(f"Result: {label} (Trust Score: {score:.2f})\n")
    
    print("--- Using Deep Learning Pipeline ---")
    predictor_dl = FakeNewsPredictor(mode='dl')
    if predictor_dl.model:
        for t in test_texts:
            label, score = predictor_dl.predict(t)
            print(f"Text Snippet: '{t[:60]}...'")
            print(f"Result: {label} (Trust Score: {score:.2f})\n")
