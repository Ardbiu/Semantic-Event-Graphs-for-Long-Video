import torch
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration
import av
import numpy as np

class VideoLLMBaseline:
    """
    Baseline 1: Strong Open-Source Video-LLM (Video-LLaVA).
    """
    def __init__(self, model_id="LanguageBind/Video-LLaVA-7B-hf", device="cuda"):
        self.device = device
        self.model_id = model_id
        
        print(f"Loading Video-LLM: {model_id} on {device}...")
        try:
            self.processor = VideoLlavaProcessor.from_pretrained(model_id)
            self.model = VideoLlavaForConditionalGeneration.from_pretrained(
                model_id, 
                torch_dtype=torch.float16 if device != 'cpu' else torch.float32,
                device_map="auto" if device == "cuda" else None
            )
            if device != "cuda" and device != "auto":
                 self.model.to(device)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load Video-LLM: {e}")
            self.model = None

    def read_video_pyav(self, container, indices):
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])

    def sample_frames(self, video_path, num_frames=8):
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        if total_frames <= 0:
            # Fallback if unknown
             total_frames = 100 
             
        indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
        clip = self.read_video_pyav(container, indices)
        return clip

    def answer_question(self, video_path, question):
        if not self.model:
            return "Error: Model not loaded."

        # 1. Sample Frames
        try:
            video_clip = self.sample_frames(video_path, num_frames=8) # 8 frames is standard for Video-LLaVA
        except Exception as e:
            return f"Error reading video: {e}"

        # 2. Prepare Prompt
        prompt = f"USER: <video>\n{question} ASSISTANT:"
        
        inputs = self.processor(text=prompt, videos=video_clip, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # 3. Generate
        with torch.no_grad():
            generate_ids = self.model.generate(**inputs, max_new_tokens=200)
            
        answer = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        # Clean up "USER: ... ASSISTANT: "
        if "ASSISTANT:" in answer:
            answer = answer.split("ASSISTANT:")[-1].strip()
            
        return answer
