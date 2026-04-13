from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from googlesearch import search
from dotenv import load_dotenv
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import json
import re

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_SEQ_LENGTH = 500

# Load Deep Learning Models
try:
    dl_model = tf.keras.models.load_model('dl_model.keras')
    with open('dl_tokenizer.pkl', 'rb') as f:
        dl_tokenizer = pickle.load(f)
    print("LSTM DL Models loaded successfully.")
except Exception as e:
    print(f"Warning: LSTM DL Models not loaded. {e}")
    dl_model = None
    dl_tokenizer = None

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
import warnings
warnings.filterwarnings("ignore") 
if api_key and len(api_key) > 5 and not api_key.startswith('your_'):
    genai.configure(api_key=api_key)
    has_gemini = True
else:
    has_gemini = False

class AnalyzeRequest(BaseModel):
    text: str = ""
    url: str = ""
    image: str = "" # Base64 DataURL

def extract_url_text(url: str) -> str:
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(r.text, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
            tag.decompose()
        return " ".join(soup.stripped_strings)[:4000]
    except Exception as e:
        print(f"URL extraction failed: {e}")
        return ""

def extract_image_ocr(image_base64: str) -> str:
    if not has_gemini: return ""
    try:
        header, encoded = image_base64.split(",", 1)
        mime = header.split(":")[1].split(";")[0]
        model = genai.GenerativeModel('gemini-2.5-flash')
        image_part = {"mime_type": mime, "data": encoded}
        prompt = "Extract ALL the legible text from this image exactly as written. Provide NO commentary, ONLY the extracted text. If no text is found, return nothing."
        extracted = model.generate_content([prompt, image_part]).text.strip()
        print(f"Extracted {len(extracted)} characters via OCR.")
        return extracted
    except Exception as e:
        print(f"Gemini OCR extraction failed: {e}")
        return ""

TRUSTED_DOMAINS = "site:reuters.com OR site:bbc.co.uk OR site:bbc.com OR site:apnews.com OR site:who.int OR site:snopes.com OR site:politifact.com OR site:npr.org site:gov OR site:gov.in"

def extract_keywords_for_search(text: str) -> str:
    if not has_gemini: return text[:50]
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        p = f"Extract a 4 to 6 word search query to verify the main factual claim in this text. Output NOTHING else except the search query:\n\n{text[:1500]}"
        return model.generate_content(p).text.strip().replace('"', '')
    except:
        return text[:50]

def search_trusted_sources(query: str):
    search_query = f"{query} {TRUSTED_DOMAINS}"
    print(f"SEARCHING INTERNET: {search_query}")
    results_text = ""
    urls = []
    try:
        results = search(search_query, num_results=3, advanced=True)
        count = 1
        for r in results:
            results_text += f"\nSOURCE {count}:\nURL: {r.url}\nTITLE: {r.title}\nSNIPPET: {r.description}\n"
            urls.append(r.url)
            count += 1
    except Exception as e:
        print(f"Internet search error: {e}")
    return results_text, urls

@app.post("/api/ml/analyze")
async def analyze_news(req: AnalyzeRequest):
    content = req.text
    
    if req.url and len(content) < 20: content = extract_url_text(req.url)
    if req.image and req.image.startswith("data:image"):
        ocr_text = extract_image_ocr(req.image)
        if len(ocr_text) > 15: content += " " + ocr_text

    if len(content) < 15:
        if has_gemini and req.image: return {"error": "Could not extract sufficient text from the image."}
        return {"error": "Text is too short for analysis."}
        
    local_prob_str = "Unavailable"
    if dl_model and dl_tokenizer:
        try:
            seq = dl_tokenizer.texts_to_sequences([content])
            X_pad = pad_sequences(seq, maxlen=MAX_SEQ_LENGTH, padding='post', truncating='post')
            prob = float(dl_model.predict(X_pad, verbose=0)[0][0])
            local_prob_str = f"{int(prob * 100)}%"
        except Exception as e:
            print(f"LSTM error: {e}")
            
    # Stage 1: Internet Verify Search
    if has_gemini:
        keywords = extract_keywords_for_search(content)
        search_snippets, extracted_urls = search_trusted_sources(keywords)
        
        # Stage 2: Final Orchestration
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = f"""
You are an AI-powered fact verification system that checks information on the internet.
Examine this user input:
"{content[:2500]}"

Here are the live internet search verification results gathered strictly from Official/Trusted Sources (BBC, Reuters, WHO, etc):
{search_snippets if search_snippets else "(No results returned from internet search)"}

And here is the Local ML Neural Network probability metric that this text's structural linguistic patterns are real: {local_prob_str}

Compare the information and analyze source credibility. Check if the claim contradicts verified reports or lacks reliable references.
If internet results are limited, lean on the ML pattern metric, mark the result as "Suspicious", and reduce the confidence score.

Output ONLY valid JSON perfectly formatted like this:
{{
  "result": "True" | "False" | "Partially True" | "Suspicious",
  "trust_score": <number 0-100 indicating trust>,
  "risk_level": "Low" | "Medium" | "High",
  "verified_sources": <array of strings exactly matching the URLs provided in the search snippets>,
  "fact_summary": "Summary of verification results, 2-3 sentences",
  "confidence": <number 0-100>
}}
DO NOT wrap in ```json. Just raw valid JSON.
"""
        try:
            response = model.generate_content(prompt)
            txt = response.text.strip()
            txt = re.sub(r'```json|```', '', txt).strip()
            
            try:
                return json.loads(txt)
            except Exception as e:
                print("Failed to parse orchestrator JSON:", txt)
                return {"error": "Failed to parse fact verification JSON."}
        except Exception as e:
            print(f"Gemini API Error: {e}")
            return {"error": "Google Gemini Rate Limit Exceeded! Please wait a minute and try again."}
            
    else:
        return {"error": "Google Gemini API required for internet fact-checking orchestration. Please add GEMINI_API_KEY to your .env file."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
