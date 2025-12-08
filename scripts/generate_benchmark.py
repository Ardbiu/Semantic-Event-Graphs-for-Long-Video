import google.generativeai as genai
import glob
import json
import os
import time

# Setup API
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("WARNING: GEMINI_API_KEY environment variable not found. Benchmark generation relies on it.")

def configure_genai():
    if api_key:
        genai.configure(api_key=api_key)

def get_model():
    model_name = 'gemini-2.5-flash'
    try:
        return genai.GenerativeModel(model_name)
    except:
        print(f"Model {model_name} not available, trying 1.5-flash")
        return genai.GenerativeModel('gemini-1.5-flash')

def generate_benchmark():
    if not api_key:
        return

    configure_genai()
    model = get_model()
    
    log_dir = os.path.join(os.path.dirname(__file__), "../outputs/logs")
    log_files = glob.glob(os.path.join(log_dir, "*_events.json"))
    if not log_files:
        print(f"No log files found in {log_dir}. Run batch_process.py first.")
        return

    benchmark_data = []
    
    for idx, log_path in enumerate(log_files):
        filename = os.path.basename(log_path)
        print(f"Generating questions for {filename} ({idx+1}/{len(log_files)})...")
        
        with open(log_path, 'r') as f:
            events = json.load(f)
            
        # Convert events to string (narrative)
        # Limit size if necessary, but 2.5-flash handles large context well.
        narrative = json.dumps(events, indent=1) 
        
        prompt = """
        Read this video event log. Generate 10 difficult questions that require connecting an event from the start of the timeline to an event at the end (long-term temporal reasoning). 
        
        Requirements:
        1. Questions must be specific.
        2. Questions must rely on the timestamp or sequence of events.
        3. Provide the correct answer based strictly on the log.
        4. Format the output strictly as a JSON list of objects:
        [
            {"q": "Question text...", "a": "Answer text..."}
        ]
        Do not output markdown code blocks, just the raw JSON.
        """
        
        full_prompt = f"{prompt}\n\nEvent Log:\n{narrative}"
        
        try:
            response = model.generate_content(full_prompt)
            text_response = response.text.strip()
            
            # Clean up markdown code blocks if the model ignores instruction (common safety)
            if text_response.startswith("```json"):
                text_response = text_response[7:]
            if text_response.endswith("```"):
                text_response = text_response[:-3]
            
            qa_pairs = json.loads(text_response)
            
            benchmark_data.append({
                "source_log": filename,
                "qa_pairs": qa_pairs
            })
            
            print(f"  Generated {len(qa_pairs)} questions.")
            
            # Rate limit politeness
            time.sleep(2)
            
        except Exception as e:
            print(f"  Error generating benchmark for {filename}: {e}")

    # Save Benchmark
    output_file = os.path.join(os.path.dirname(__file__), "../outputs/benchmark_data.json")
    with open(output_file, "w") as f:
        json.dump(benchmark_data, f, indent=2)
    
    print(f"\nBenchmark generation complete. Saved to {output_file}")

if __name__ == "__main__":
    generate_benchmark()
