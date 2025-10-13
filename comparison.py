import os
import json
import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util

BASELINE_OUTPUT_DIR = "baseline_outputs"
LLM_OUTPUT_DIR = "output"
DATASET_DIR = "dataset"
COMPARISON_RESULT_FILE = "semantic_comparison_results.csv"

print("Loading Sentence Transformer model for semantic analysis...")
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")


def normalize_and_flatten(keywords_list):
    """Takes a list of phrases, splits them into words, lowercases, and returns a unique set."""
    words = set()
    for phrase in keywords_list:
        cleaned_phrase = re.sub(r'[^a-zA-Z0-9\s]', '', phrase.lower())
        for word in cleaned_phrase.split():
            words.add(word)
    return words

def calculate_jaccard_similarity(set1, set2):
    """Calculates the Jaccard Similarity between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def calculate_avg_phrase_length(phrases):
    """Calculates the average number of words in a list of phrases."""
    if not phrases:
        return 0
    total_words = sum(len(phrase.split()) for phrase in phrases)
    return total_words / len(phrases)

def calculate_semantic_similarity(model, text, keywords):
    """Calculates the cosine similarity between a text and a list of keywords."""
    if not keywords:
        return 0
    
    keyword_sentence = ". ".join(keywords)
    
    text_embedding = model.encode(text, convert_to_tensor=True)
    keyword_embedding = model.encode(keyword_sentence, convert_to_tensor=True)
    
    cosine_score = util.cos_sim(text_embedding, keyword_embedding)
    return cosine_score.item()

def main():
    baseline_path = os.path.join(BASELINE_OUTPUT_DIR, "baseline_keywords.json")
    with open(baseline_path, 'r', encoding='utf-8') as f:
        baseline_data = json.load(f)
    baseline_lookup = {item['file']: item for item in baseline_data}
    
    comparison_results = []

    print("Starting comparison of baseline vs. LLM outputs...")
    llm_files = [f for f in os.listdir(LLM_OUTPUT_DIR) if f.endswith("_medium_low.json")]

    for llm_filename in llm_files:
        original_filename = llm_filename.replace("_medium_low.json", ".txt")
        if original_filename not in baseline_lookup:
            continue
            
        with open(os.path.join(LLM_OUTPUT_DIR, llm_filename), 'r', encoding='utf-8') as f:
            llm_data = json.load(f)
        baseline_item = baseline_lookup[original_filename]
        
        with open(os.path.join(DATASET_DIR, original_filename), 'r', encoding='utf-8') as f:
            source_text = f.read()
        
        baseline_all_keywords = []
        for key in ["rake", "tfidf", "keybert", "noun_chunks"]:
            baseline_all_keywords.extend(baseline_item.get(key, []))

        llm_all_keywords = []
        for key in ["primary_keywords", "secondary_keywords", "key_phrases"]:
             llm_all_keywords.extend(llm_data.get(key, []))
        
        jaccard_score = calculate_jaccard_similarity(
            normalize_and_flatten(baseline_all_keywords),
            normalize_and_flatten(llm_all_keywords)
        )
        avg_len_llm = calculate_avg_phrase_length(llm_all_keywords)
        avg_len_baseline = calculate_avg_phrase_length(baseline_all_keywords)
        
        semantic_sim_baseline = calculate_semantic_similarity(similarity_model, source_text, baseline_all_keywords)
        semantic_sim_llm = calculate_semantic_similarity(similarity_model, source_text, llm_all_keywords)
        
        comparison_results.append({
            "file": original_filename,
            "jaccard_similarity": round(jaccard_score, 4),
            "avg_words_baseline": round(avg_len_baseline, 2),
            "avg_words_llm": round(avg_len_llm, 2),
            "semantic_sim_baseline": round(semantic_sim_baseline, 4),
            "semantic_sim_llm": round(semantic_sim_llm, 4)
        })

    df = pd.DataFrame(comparison_results)
    df.to_csv(COMPARISON_RESULT_FILE, index=False)
    
    print(f"\nComparison complete. Results saved to '{COMPARISON_RESULT_FILE}'")
    print("\n--- Sample Results ---")
    print(df.head().to_string())

if __name__ == "__main__":
    main()