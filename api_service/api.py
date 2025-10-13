import os
import json
import re
import nltk
from nltk.corpus import stopwords
from flask import Flask, request, jsonify
import requests

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

from rake_nltk import Rake
from sklearn.feature_extraction.text import TfidfVectorizer
from keybert import KeyBERT
import spacy

OLLAMA_API_URL = "http://ollama:11434/api/chat"
MODEL_TO_USE = "mistral:latest" 
TOP_N_KEYWORDS = 15
STOPWORDS = set(stopwords.words('english'))

app = Flask(__name__)

print("Loading NLP models for baseline analysis...")
nlp = spacy.load("en_core_web_sm")
kw_model = KeyBERT()
print("Models loaded successfully.")


def analyze_text_with_llm(text_content: str) -> dict | None:
    prompt = f"""
    Analyze the following text and extract the required information.
    **Strictly adhere to these rules:**
    1. All keywords, phrases, and sentences MUST be extracted directly from the text.
    2. Your response MUST be only a single valid JSON object, with no other text before or after it.
    --- TEXT TO ANALYZE ---
    {text_content}
    ---
    --- REQUIRED JSON STRUCTURE ---
    {{
      "primary_keywords": ["..."], "secondary_keywords": ["..."], "key_phrases": ["..."],
      "long_tail_phrases": ["..."], "evidence_sentences": ["..."], "confidence": 0.0, "must_include": []
    }}
    """
    payload = {
        "model": MODEL_TO_USE, "format": "json", "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0.2}
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=180)
        response.raise_for_status()
        response_data = response.json()
        return json.loads(response_data['message']['content'])
    except Exception as e:
        print(f"LLM analysis failed: {e}")
        return {"error": f"LLM analysis failed: {e}"}

def preprocess_text(text):
    text = text.lower()
    text = text.replace('\n', ' ')
    text = re.sub(' +', ' ', text)
    return text

def extract_rake(text):
    rake = Rake(min_length=2, max_length=4)
    rake.extract_keywords_from_text(text)
    return rake.get_ranked_phrases()[:TOP_N_KEYWORDS]

def extract_tfidf_for_single_doc(text):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    try:
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray().flatten()
        top_indices = scores.argsort()[::-1][:TOP_N_KEYWORDS]
        return [feature_names[i] for i in top_indices]
    except ValueError: return []

def extract_noun_chunks(text):
    doc = nlp(text)
    chunks = set()
    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) > 1 and chunk.text not in STOPWORDS and len(chunk.text) > 3:
            chunks.add(chunk.text)
    return list(chunks)[:TOP_N_KEYWORDS]

def extract_keybert(text):
    keywords = kw_model.extract_keywords(
        text, keyphrase_ngram_range=(1, 2), stop_words='english', 
        use_mmr=True, diversity=0.7, top_n=TOP_N_KEYWORDS
    )
    return [kw[0] for kw in keywords]

def run_baseline_analysis(text):
    """Runs all baseline methods on the text and returns a dictionary."""
    clean_text = preprocess_text(text)
    return {
        "rake": extract_rake(clean_text),
        "tfidf": extract_tfidf_for_single_doc(clean_text),
        "keybert": extract_keybert(clean_text),
        "noun_chunks": extract_noun_chunks(clean_text)
    }

@app.route("/extract", methods=["POST"])
def extract_keywords():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
        
    data = request.get_json()
    text = data.get("text", None)
    
    if not text:
        return jsonify({"error": "Missing 'text' key in request body"}), 400
        
    llm_result = analyze_text_with_llm(text)
    baseline_result = run_baseline_analysis(text)
    
    combined_result = {
        "llm_analysis": llm_result,
        "baseline_analysis": baseline_result
    }
        
    return jsonify(combined_result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)