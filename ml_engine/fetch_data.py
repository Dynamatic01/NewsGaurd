import pandas as pd
import numpy as np
import random
import os

DATA_FILE = "fake_or_real_news.csv"

def generate_synthetic_data(num_samples=1000):
    print(f"Generating synthetic fake news dataset ({num_samples} samples)...")
    
    real_keywords = ["government", "policy", "economy", "health", "announced", "officials", "report", "growth", "international", "agreement", "study", "research", "technology", "election"]
    fake_keywords = ["shocking", "alien", "conspiracy", "secret", "exposed", "hoax", "banned", "cure", "miracle", "hidden", "destroyed", "illegal", "truth", "mind control", "lizard"]
    
    data = []
    for i in range(num_samples):
        # 0 = FAKE, 1 = REAL
        is_real = random.choice([0, 1])
        
        words = []
        # Generate slightly realistic sounding gibberish
        if is_real:
            base_words = random.choices(real_keywords, k=random.randint(20, 50))
            words.extend(base_words)
            label = "REAL"
        else:
            base_words = random.choices(fake_keywords, k=random.randint(20, 50))
            words.extend(base_words)
            label = "FAKE"
            
        # Add some random generic words to both so the model has to learn
        generic = ["the", "and", "they", "will", "be", "in", "to", "of", "a", "that", "this"]
        words.extend(random.choices(generic, k=random.randint(10, 30)))
        random.shuffle(words)
        
        text = " ".join(words).capitalize() + "."
        title = " ".join(words[:5]).capitalize()
        
        data.append({"id": i, "title": title, "text": text, "label": label})
        
    df = pd.DataFrame(data)
    df.to_csv(DATA_FILE, index=False)
    print(f"Data successfully saved to {DATA_FILE}")
    print(df.head(3))
    return df

if __name__ == "__main__":
    generate_synthetic_data()
