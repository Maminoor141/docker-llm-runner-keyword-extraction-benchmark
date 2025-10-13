import os
import json
import re
import nltk
from nltk.corpus import stopwords
import concurrent.futures
import time

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

from rake_nltk import Rake
from sklearn.feature_extraction.text import TfidfVectorizer
from keybert import KeyBERT
import spacy

DATASET_DIR = "dataset"
OUTPUT_DIR = "baseline_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TOP_N_KEYWORDS = 15
MAX_WORKERS = 4
STOPWORDS = set(stopwords.words('english'))

print("Loading NLP models, this may take a moment...")
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Spacy model 'en_core_web_sm' not found. Downloading...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    
kw_model = KeyBERT()
print("Models loaded successfully.")

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
    except ValueError:
        return []

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

def process_file(filename):
    """Reads a file, processes it, and returns the result dictionary."""
    path = os.path.join(DATASET_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    clean_text = preprocess_text(raw_text)
    
    page_result = {
        "file": filename,
        "rake": extract_rake(clean_text),
        "tfidf": extract_tfidf_for_single_doc(clean_text),
        "keybert": extract_keybert(clean_text),
        "noun_chunks": extract_noun_chunks(clean_text)
    }
    print(f"Finished processing: {filename}")
    return page_result

def main():
    start_time = time.time()

    files_to_process = sorted([f for f in os.listdir(DATASET_DIR) if f.endswith(".txt")])
    
    all_results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results_iterator = executor.map(process_file, files_to_process)
        all_results = list(results_iterator)

    all_results = [res for res in all_results if res is not None]
    all_results.sort(key=lambda x: x['file'])

    output_path = os.path.join(OUTPUT_DIR, "baseline_keywords.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\nBaseline keyword extraction complete. Results saved in '{output_path}'")
    print(f"Total execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()