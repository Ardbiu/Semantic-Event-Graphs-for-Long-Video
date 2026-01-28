import numpy as np
from sentence_transformers import SentenceTransformer, util
import math
from typing import Dict, Tuple, List, Optional

class InteractionScorer:
    """
    Computes the interaction score S(i,j,t) between two objects.
    S = w1*Proximity + w2*Motion + w3*Semantic + w4*Action
    """
    def __init__(self, config: dict):
        self.config = config
        self.weights = config.get('weights', {
            'w1_proximity': 0.4,
            'w2_motion': 0.3,
            'w3_semantic': 0.3,
            'w4_action': 0.0
        })
        self.distance_threshold = config.get('distance_threshold', 200.0)
        
        # Initialize semantic model if enabled
        self.use_semantic = config.get('semantic', {}).get('use_embedding', False)
        self.semantic_model = None
        self.embedding_cache = {}
        
        if self.use_semantic:
            model_name = config.get('semantic', {}).get('embedding_model', 'all-MiniLM-L6-v2')
            try:
                print(f"Loading semantic model: {model_name}...")
                self.semantic_model = SentenceTransformer(model_name)
            except Exception as e:
                print(f"Warning: Failed to load semantic model {model_name}: {e}. Disabling semantic component.")
                self.use_semantic = False

    def compute_score(self, det_a: dict, det_b: dict, 
                     prev_det_a: Optional[dict] = None, prev_det_b: Optional[dict] = None, 
                     fps: float = 30.0) -> float:
        
        # 1. Proximity Score (normalized)
        prox_score = self._compute_proximity(det_a, det_b)
        
        # 2. Motion Coupling Score
        motion_score = self._compute_motion_coupling(det_a, det_b, prev_det_a, prev_det_b, fps)
        
        # 3. Semantic Compatibility Score
        sem_score = self._compute_semantic_compatibility(det_a['label'], det_b['label'])
        
        # 4. Action Cue (Placeholder)
        action_score = 0.0
        
        # Final Weighted Score
        S = (self.weights['w1_proximity'] * prox_score +
             self.weights['w2_motion'] * motion_score +
             self.weights['w3_semantic'] * sem_score +
             self.weights['w4_action'] * action_score)
             
        return S, {'prox': prox_score, 'motion': motion_score, 'sem': sem_score}

    def _compute_centroid(self, bbox) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def _compute_proximity(self, det_a, det_b) -> float:
        """
        Returns 1.0 if distance is 0, decays to 0.0 at distance_threshold.
        Uses a soft decay function: max(0, 1 - dist/threshold)
        """
        c_a = self._compute_centroid(det_a['bbox'])
        c_b = self._compute_centroid(det_b['bbox'])
        dist = math.hypot(c_a[0] - c_b[0], c_a[1] - c_b[1])
        
        if dist >= self.distance_threshold:
            return 0.0
        return 1.0 - (dist / self.distance_threshold)

    def _compute_motion_coupling(self, det_a, det_b, prev_det_a, prev_det_b, fps) -> float:
        """
        Correlates velocity vectors. If both are moving effectively and their relative distance 
        is changing or they are moving together, score is higher.
        Simplified: Cosine similarity of velocity vectors if both moving.
        """
        if prev_det_a is None or prev_det_b is None:
            return 0.0
            
        # Current centroids
        ca_curr = self._compute_centroid(det_a['bbox'])
        cb_curr = self._compute_centroid(det_b['bbox'])
        
        # Previous centroids
        ca_prev = self._compute_centroid(prev_det_a['bbox'])
        cb_prev = self._compute_centroid(prev_det_b['bbox'])
        
        # Velocity vectors
        v_a = np.array([ca_curr[0] - ca_prev[0], ca_curr[1] - ca_prev[1]])
        v_b = np.array([cb_curr[0] - cb_prev[0], cb_curr[1] - cb_prev[1]])
        
        # Magnitudes
        mag_a = np.linalg.norm(v_a)
        mag_b = np.linalg.norm(v_b)
        
        # Threshold for "moving" (e.g., 2 pixels per frame)
        min_move = 2.0
        
        if mag_a < min_move or mag_b < min_move:
            return 0.0
            
        # Cosine similarity
        cos_sim = np.dot(v_a, v_b) / (mag_a * mag_b + 1e-6)
        
        # Map cosine [-1, 1] to [0, 1] loosely? 
        # Actually, for interaction:
        # - Moving towards each other (converging) might be interaction start (-1 similarity if head on?)
        # - Moving together (parallel) might be "walking with" (+1 similarity)
        # Let's check RELATIVE velocity. 
        # High relative velocity approaching = interaction imminent.
        # Low relative velocity while close = sustained interaction (handled by proximity).
        
        # Let's stick to the paper plan: "Korrelate velocity vectors"
        # We'll use absolute cosine similarity for now: |cos_sim|
        # Or simply return 0.5 if moving at all, 1.0 if correlated.
        
        # Simple heuristic: if moving close to each other, add bonus.
        return (cos_sim + 1.0) / 2.0  # Normalized to 0-1

    def _compute_semantic_compatibility(self, label_a: str, label_b: str) -> float:
        if not self.use_semantic or not self.semantic_model:
            # Fallback to simple rule or 0.5
            return 0.5
            
        key = tuple(sorted((label_a, label_b)))
        if key in self.embedding_cache:
            return self.embedding_cache[key]
            
        # Compute embedding similarity
        emb_a = self.semantic_model.encode(label_a, convert_to_tensor=True)
        emb_b = self.semantic_model.encode(label_b, convert_to_tensor=True)
        
        score = util.pytorch_cos_sim(emb_a, emb_b).item()
        
        # Cache it
        self.embedding_cache[key] = max(0.0, score) # Ensure non-negative
        return self.embedding_cache[key]
