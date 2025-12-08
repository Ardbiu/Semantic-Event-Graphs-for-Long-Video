import google.generativeai as genai
import os
import json

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
    log_path = os.path.join(os.path.dirname(__file__), "../outputs/event_log.json")
    if not os.path.exists(log_path):
        # try legacy path
        log_path = os.path.join(os.path.dirname(__file__), "../data/event_log.json")

    with open(log_path, "r") as f:
        events = json.load(f)
        
    # 2. Filter Last 30 Seconds
    # Find max timestamp
    if not events:
        print("Event log is empty.")
        return

    max_time = max(ev.get("timestamp", 0) for ev in events)
    cutoff_time = max_time - 30.0
    
    recent_events = [ev for ev in events if ev.get("timestamp", 0) >= cutoff_time]
    
    narrative = events_to_narrative(recent_events)
    
    # 3. Prepare Prompt
    question = "When did he use the laptop?"
    prompt = f"""
    Based on the following video event log, answer the question. 
    If the information is not in the log, say "I cannot determine the answer from the context."
    
    Event Log (Last 30 seconds):
    {narrative}
    
    Question: {question}
    """
    
    model = get_model()
    
    print(f"Processing baseline_short with {len(recent_events)} events...")
    
    try:
        response = model.generate_content(prompt)
        answer = response.text
    except Exception as e:
        answer = f"Error: {e}"
        
    # Check for success (heuristic)
    found_info = "laptop" in answer.lower() and "cannot determine" not in answer.lower()
    
    print("\n--- Baseline Short Results ---")
    print(f"Events Sent: {len(recent_events)}")
    print(f"Answer: {answer}")
    print(f"Successfully Found Info: {found_info}")

if __name__ == "__main__":
    main()
