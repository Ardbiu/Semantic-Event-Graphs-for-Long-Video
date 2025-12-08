import google.generativeai as genai
import os
import json
import time

# Setup API
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("WARNING: GEMINI_API_KEY environment variable not found.")
else:
    genai.configure(api_key=api_key)

MODEL_NAME = 'gemini-2.5-flash'

def get_model():
    try:
        return genai.GenerativeModel(MODEL_NAME)
    except:
        print(f"Model {MODEL_NAME} not found, falling back to gemini-1.5-flash")
        return genai.GenerativeModel('gemini-1.5-flash')

def events_to_narrative(events):
    narrative = []
    for ev in events:
        line = f"[{ev.get('timestamp', 0):.1f}s] {ev.get('subject')} {ev.get('type')} {ev.get('object')}"
        narrative.append(line)
    return "\n".join(narrative)

def main():
    if not api_key:
        return

    # 1. Load Data
    with open("event_log.json", "r") as f:
        events = json.load(f)
    
    # 2. Convert to Narrative
    narrative = events_to_narrative(events)
    
    # 3. Prepare Prompt
    question = "Where is the laptop?"
    prompt = f"""
    Based on the following video event log, answer the question.
    
    Event Log:
    {narrative}
    
    Question: {question}
    """
    
    model = get_model()
    
    # 4. Count Tokens
    input_tokens = model.count_tokens(prompt).total_tokens
    
    print(f"Processing baseline_long with {input_tokens} input tokens...")
    
    # 5. Measure Time and Generate
    start_time = time.time()
    try:
        response = model.generate_content(prompt)
        answer = response.text
    except Exception as e:
        answer = f"Error: {e}"
    end_time = time.time()
    
    duration = end_time - start_time
    
    print("\n--- Baseline Long Results ---")
    print(f"Time Taken: {duration:.2f} seconds")
    print(f"Input Tokens: {input_tokens}")
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
