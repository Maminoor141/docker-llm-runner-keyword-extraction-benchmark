import os
import json
import requests
import concurrent.futures
import time

# --- Configuration ---
# OLLAMA_API_URL = "http://host.docker.internal:11434/api/chat"
# OLLAMA_API_URL = "http://localhost:11434/api/chat"
OLLAMA_API_URL = "http://ollama:11434/api/chat"
MODEL_TO_USE = "mistral:latest" 
DATASET_FOLDER = "dataset"
OUTPUT_FOLDER = "output"
MAX_WORKERS = 4

def analyze_text_with_llm(text_content: str, generation_options: dict) -> dict | None:
    """Sends text content to the Ollama API with specific generation settings."""
    prompt = f"""
    Analyze the following text and extract the required information.
    **Strictly adhere to these rules:**
    1. All keywords, phrases, and sentences MUST be extracted directly from the text.
    2. The 'must_include' list should contain terms from headings. If there are no headings, it MUST be an empty list.
    3. Your response MUST be only a single valid JSON object, with no other text before or after it.

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
        "messages": [{"role": "user", "content": prompt}], "options": generation_options
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=180)
        response.raise_for_status()
        response_data = response.json()
        return json.loads(response_data['message']['content'])
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {e}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}

def process_file(filename):
    """
    Reads a single file and calls the LLM API for both medium-low and high settings.
    Saves the results to their respective files.
    """
    print(f"Starting processing for: {filename}")
    filepath = os.path.join(DATASET_FOLDER, filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        if not content.strip():
            print(f"File is empty, skipping: {filename}")
            return f"Skipped empty file: {filename}"
            
        settings_to_test = [
            ("medium_low", {"temperature": 0.3, "top_p": 1.0}),
            ("high", {"temperature": 0.9, "top_p": 0.9})
        ]

        for setting_name, generation_options in settings_to_test:
            extracted_data = analyze_text_with_llm(content, generation_options)
            
            if extracted_data and "error" not in extracted_data:
                base_filename, _ = os.path.splitext(filename)
                output_filename = f"{base_filename}_{setting_name}.json"
                output_filepath = os.path.join(OUTPUT_FOLDER, output_filename)
                with open(output_filepath, 'w', encoding='utf-8') as f:
                    json.dump(extracted_data, f, indent=4)
            elif extracted_data and "error" in extracted_data:
                 print(f"Error processing {filename} with {setting_name} settings: {extracted_data['error']}")

        print(f"Finished processing: {filename}")
        return f"Successfully processed: {filename}"

    except Exception as e:
        print(f"Failed to read or process file {filename}. Error: {e}")
        return f"Failed: {filename}"

def main():
    """
    Main function to process files in parallel using a ThreadPoolExecutor.
    """
    start_time = time.time()

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    files_to_process = sorted([f for f in os.listdir(DATASET_FOLDER) if f.endswith(".txt")])
    if not files_to_process:
        print(f"No .txt files found in the '{DATASET_FOLDER}' directory.")
        return

    print(f"Found {len(files_to_process)} text files. Starting parallel processing...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(process_file, files_to_process))

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\nAll done! Check the '{OUTPUT_FOLDER}' directory for your JSON files.")
    success_count = sum(1 for r in results if r and r.startswith("Successfully"))
    print(f"Successfully processed {success_count} out of {len(files_to_process)} files.")
    print(f"Total execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()