from typing import Dict, Tuple

class FusionEngine:
    def __init__(self):
        # Default Weights (can be tuned)
        self.default_face_weight = 0.6
        self.default_voice_weight = 0.4
        
        # Uncertainty Rule
        self.face_confidence_threshold = 0.6
        
        # Post-threshold weights
        self.uncertain_face_weight = 0.2
        self.uncertain_voice_weight = 0.8

    def _get_confidence(self, emotions: Dict[str, float]) -> float:
        """Returns the maximum probability score as the confidence metric."""
        if not emotions:
            return 0.0
        return max(emotions.values())

    def fuse(self, face_result: Dict[str, float], voice_result: Dict[str, float]) -> Tuple[Dict[str, float], Dict]:
        """
        Fuses face and voice emotion probabilities.
        Returns:
            - Fused probabilities (Dict)
            - Decision log (Dict) containing weights used and reasoning.
        """
        face_conf = self._get_confidence(face_result)
        voice_conf = self._get_confidence(voice_result)
        
        # Dynamic Weighting Logic
        if face_conf < self.face_confidence_threshold:
            w_face = self.uncertain_face_weight
            w_voice = self.uncertain_voice_weight
            reason = f"Face confidence ({face_conf:.2f}) < {self.face_confidence_threshold}. Prioritizing Voice."
        else:
            w_face = self.default_face_weight
            w_voice = self.default_voice_weight
            reason = "Face confidence sufficient. Using standard weights."

        # Merge emotion keys (union of all detected emotions)
        all_emotions = set(face_result.keys()) | set(voice_result.keys())
        fused_scores = {}
        
        for emotion in all_emotions:
            p_face = face_result.get(emotion, 0.0)
            p_voice = voice_result.get(emotion, 0.0)
            
            # Weighted Sum Fusion
            fused_scores[emotion] = (p_face * w_face) + (p_voice * w_voice)
        
        # Normalize scores to sum to 1.0 (optional, but good practice)
        total_score = sum(fused_scores.values())
        if total_score > 0:
            for k in fused_scores:
                fused_scores[k] /= total_score
        
        log = {
            "face_confidence": face_conf,
            "voice_confidence": voice_conf,
            "weights_used": {"face": w_face, "voice": w_voice},
            "reasoning": reason
        }
        
        return fused_scores, log
