import google.generativeai as genai
import os

class SummarizationBaseline:
    """
    Baseline R2: Uniform clip sampling -> Summarize -> Answer with LLM.
    Uses Gemini for both summarization and answering for efficiency.
    """
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.model = None

    def answer_question(self, video_path, question):
        if not self.model:
            return "Error: API Key missing."
        
        # Upload video (Gemini handles the sampling/processing internally effectively)
        # This is strictly "Summarization" if we prompt it to summarize first.
        
        try:
            print("Uploading video to Gemini...")
            video_file = genai.upload_file(video_path)
            
            # Wait for processing
            import time
            while video_file.state.name == "PROCESSING":
                time.sleep(2)
                video_file = genai.get_file(video_file.name)
                
            if video_file.state.name == "FAILED":
                return "Error: Video processing failed."
                
            # Chain: Summarize -> Answer
            # Prompt 1: Summarize
            prompt_sum = "Summarize this video in detail, focusing on all interactions and objects."
            response_sum = self.model.generate_content([video_file, prompt_sum])
            summary = response_sum.text
            
            # Prompt 2: Answer from Summary
            prompt_qa = f"Based on this summary of a video, answer the question.\n\nSummary: {summary}\n\nQuestion: {question}"
            response_qa = self.model.generate_content(prompt_qa)
            
            return response_qa.text.strip()
            
        except Exception as e:
            return f"Error: {e}"
