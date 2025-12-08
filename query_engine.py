import google.generativeai as genai
import os
import json
from temporal_graph import TemporalSceneGraph

# Setup API
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    # If the user forgot to export it, this will raise an error or print a warning
    print("WARNING: GEMINI_API_KEY environment variable not found. Please export it.")
else:
    genai.configure(api_key=api_key)

# Model Selection
# Fallback logic if 2.5 is not available: [m.name for m in genai.list_models()]
# We will assume 'gemini-2.0-flash-exp' or similar if 2.5 is strictly required by the prompt 
# but the prompt asked for "gemini-2.5-flash". I will use that string.
# Note: As of my knowledge cutoff, 1.5 is the latest major public. The user prompt explicitly requested 2.5.
# I will use it as requested, assuming user has access.
MODEL_NAME = 'gemini-2.5-flash' 
try:
    model = genai.GenerativeModel(MODEL_NAME)
except Exception as e:
    print(f"Error initializing model '{MODEL_NAME}': {e}")
    print("Available models:", [m.name for m in genai.list_models()])
    # Fallback/Suggestion
    model = genai.GenerativeModel('gemini-1.5-flash') # Safer default if 2.5 doesn't work for me
    print("Falling back to 'gemini-1.5-flash' for demonstration if needed.")


def events_to_narrative(events):
    """Converts a list of events into a textual narrative."""
    narrative = []
    for ev in events:
        # Construct a readable line: [timestamp] Subject Interaction Object
        line = f"[{ev.get('timestamp', 0):.1f}s] {ev.get('subject')} {ev.get('type')} {ev.get('object')}"
        narrative.append(line)
    return "\n".join(narrative)

def main():
    # 1. Load Data
    tsg = TemporalSceneGraph()
    # Check for event_log.json, if not exists, create dummy (reuse logic from temporal_graph for robustness)
    if not os.path.exists("event_log.json"):
        dummy_data = [
            {"timestamp": 120.5, "frame": 3600, "type": "START", "subject": "person-1", "object": "coffee_cup-3"},
            {"timestamp": 125.0, "frame": 3750, "type": "END", "subject": "person-1", "object": "coffee_cup-3"},
            {"timestamp": 400.2, "frame": 12000, "type": "START", "subject": "person-1", "object": "laptop-7"}
        ]
        with open("event_log.json", "w") as f:
            json.dump(dummy_data, f, indent=2)
        print("Created dummy event_log.json")
    
    tsg.load_from_json("event_log.json")
    
    # 2. User Question
    question = "Where is the laptop?"
    print(f"Question: {question}")
    
    # 3. Prune
    pruned_result = tsg.prune_and_retrieve(question)
    pruned_narrative = events_to_narrative(pruned_result.events)
    
    # Narrative for Full Log (for comparison)
    full_narrative = events_to_narrative(tsg.all_events)
    
    # 4. Token Counting & 5. Generate Answer
    if not api_key:
        print("Skipping API calls due to missing API key.")
        return

    try:
        # Count tokens
        pruned_tokens = model.count_tokens(pruned_narrative).total_tokens
        full_tokens = model.count_tokens(full_narrative).total_tokens
        
        print("\n--- Efficiency Analysis ---")
        if full_tokens > 0:
            reduction = (1 - (pruned_tokens / full_tokens)) * 100
        else:
            reduction = 0
            
        print(f"Full Context Tokens: {full_tokens}")
        print(f"Pruned Context Tokens: {pruned_tokens}")
        print(f"Efficiency Gain: Processed {pruned_tokens} tokens instead of {full_tokens} ({reduction:.1f}% reduction)")

        # Generate Content
        prompt = f"""
        Based on the following video event log, answer the question.
        
        Event Log:
        {pruned_narrative}
        
        Question: {question}
        """
        
        response = model.generate_content(prompt)
        
        print("\n--- Answer ---")
        print(response.text)
        
    except Exception as e:
        print(f"An error occurred during API interaction: {e}")

if __name__ == "__main__":
    main()
