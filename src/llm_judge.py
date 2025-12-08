import google.generativeai as genai
import os

# Setup API
api_key = os.environ.get("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    print("WARNING: GEMINI_API_KEY environment variable not found. LLM Judge will fail.")

def evaluate_accuracy(question, ground_truth, model_answer):
    """
    Uses Gemini to semantically judge if the model answer matches the ground truth.
    Returns: True if equivalent, False otherwise.
    """
    if not api_key:
        print("Error: API key missing.")
        return False

    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""
        System Instruction: You are a specialized evaluation model for long-video QA systems. Your sole task is to determine if the Model Answer is semantically equivalent to the Ground Truth Answer for the given Question. Ignore differences in phrasing, verb tense, or units (e.g., '2 minutes' vs '120 seconds'). Respond only with the boolean word 'TRUE' or 'FALSE'.

        Question: {question}
        Ground Truth: {ground_truth}
        Model Answer: {model_answer}
        """
        
        response = model.generate_content(prompt)
        text_response = response.text.strip().upper()
        
        if "TRUE" in text_response:
            return True
        elif "FALSE" in text_response:
            return False
        else:
            # Fallback for unexpected output
            print(f"Warning: Unexpected LLM Judge response: {text_response}")
            return False

    except Exception as e:
        print(f"Error calling LLM Judge: {e}")
        return False

if __name__ == "__main__":
    # Test Cases
    print("Running LLM Judge Test Cases...\n")
    
    q_test = "What does the person use to prop up the light?"
    gt_test = "His water bottle and shoes."
    
    # Case 1: Correct
    ans_correct = "A water bottle and his shoes."
    result_correct = evaluate_accuracy(q_test, gt_test, ans_correct)
    print(f"Test 1 (Expected TRUE): {result_correct}")
    
    # Case 2: Incorrect
    ans_incorrect = "A coffee cup."
    result_incorrect = evaluate_accuracy(q_test, gt_test, ans_incorrect)
    print(f"Test 2 (Expected FALSE): {result_incorrect}")
    
    # Summary
    if result_correct and not result_incorrect:
        print("\nSUCCESS: LLM Judge logic verified.")
    else:
        print("\nFAILURE: LLM Judge logic failed.")
