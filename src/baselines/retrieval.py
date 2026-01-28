import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import cv2
import numpy as np

class RetrievalBaseline:
    """
    Baseline R1: Caption frames -> Embed -> Retrieve top-k -> Answer with LLM.
    """
    def __init__(self, device="cuda"):
        self.device = device
        # 1. Captioner
        print("Loading BLIP captioner...")
        try:
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
        except Exception:
            self.caption_model = None

        # 2. Retriever
        print("Loading Sentence Transformer...")
        try:
            self.retriever = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception:
            self.retriever = None

    def caption_frames(self, video_path, fps_sample=1.0):
        captions = []
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if fps <= 0: return []
        
        step = int(fps / fps_sample)
        if step < 1: step = 1
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            if frame_idx % step == 0:
                # Caption
                inputs = self.processor(images=frame, return_tensors="pt").to(self.device)
                out = self.caption_model.generate(**inputs)
                caption = self.processor.decode(out[0], skip_special_tokens=True)
                timestamp = frame_idx / fps
                captions.append({'timestamp': timestamp, 'text': caption})
                
            frame_idx += 1
        cap.release()
        return captions

    def retrieve(self, captions, query, top_k=5):
        if not self.retriever or not captions:
            return []
            
        texts = [c['text'] for c in captions]
        
        query_emb = self.retriever.encode(query, convert_to_tensor=True)
        doc_embs = self.retriever.encode(texts, convert_to_tensor=True)
        
        scores = util.cos_sim(query_emb, doc_embs)[0]
        
        # Get top k
        top_results = torch.topk(scores, k=min(top_k, len(texts)))
        
        retrieved = []
        for score, idx in zip(top_results.values, top_results.indices):
            retrieved.append(captions[idx])
            
        # Sort by timestamp to maintain narrative flow
        retrieved.sort(key=lambda x: x['timestamp'])
        return retrieved
