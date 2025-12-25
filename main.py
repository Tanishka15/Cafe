from detectors import MockFaceEmotionDetector, MockVoiceEmotionDetector
from fusion_engine import FusionEngine
import json

def run_scenario(name, face_input, voice_input, face_det, voice_det, engine):
    print(f"\n--- Running Scenario: {name} ---")
    
    # 1. Set Mock Inputs
    face_det.set_result(face_input)
    voice_det.set_result(voice_input)
    
    # 2. Detect
    face_res = face_det.detect()
    voice_res = voice_det.detect()
    
    print(f"Face Input: {json.dumps(face_res)}")
    print(f"Voice Input: {json.dumps(voice_res)}")
    
    # 3. Fuse
    fused_result, log = engine.fuse(face_res, voice_res)
    
    # 4. Output
    print(f"Decision Log: {json.dumps(log, indent=2)}")
    print(f"Fused Result: {json.dumps(fused_result, indent=2)}")
    
    # Return for assertion/verification if needed
    return fused_result, log

def main():
    # Initialize
    face_detector = MockFaceEmotionDetector()
    voice_detector = MockVoiceEmotionDetector()
    engine = FusionEngine()
    
    # Scenario 1: Reliable Face (Face Confidence = 0.9)
    # Expected: Face dominant (0.6 vs 0.4 default weights)
    run_scenario(
        "Reliable Face",
        {"happy": 0.9, "neutral": 0.1},
        {"neutral": 0.8, "happy": 0.2},
        face_detector, voice_detector, engine
    )

    # Scenario 2: Unreliable Face (Face Confidence = 0.4)
    # Expected: Voice dominant (0.8 vs 0.2 uncertainty weights)
    # Face says Happy (low conf), Voice says Angry (high conf) -> Result should be Angry
    run_scenario(
        "Unreliable Face (< 0.6) - Trigger Weighted Switch",
        {"happy": 0.4, "sad": 0.3, "neutral": 0.3},
        {"angry": 0.8, "neutral": 0.2},
        face_detector, voice_detector, engine
    )

    # Scenario 3: Ambiguous / Conflict
    # Face: Sad (0.55) - Just below threshold usually, but let's see. 
    # Note: 0.55 < 0.6, so Voice should still dominate.
    # Voice: Happy (0.7)
    run_scenario(
        "Ambiguous Edge Case (Face 0.55)",
        {"sad": 0.55, "neutral": 0.45},
        {"happy": 0.7, "neutral": 0.3},
        face_detector, voice_detector, engine
    )

if __name__ == "__main__":
    main()
