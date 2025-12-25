from abc import ABC, abstractmethod
from typing import Dict, Optional

class EmotionDetector(ABC):
    """Abstract base class for emotion detectors."""
    
    @abstractmethod
    def detect(self) -> Dict[str, float]:
        """
        Returns a dictionary of emotion probabilities.
        """
        pass

class MockFaceEmotionDetector(EmotionDetector):
    """
    Mock Face Emotion Detector.
    Allows manual setting of the result for testing purposes.
    """
    def __init__(self):
        self._current_result: Optional[Dict[str, float]] = None

    def set_result(self, result: Dict[str, float]):
        """Sets the result to be returned by detect()."""
        self._current_result = result

    def detect(self) -> Dict[str, float]:
        if self._current_result is None:
            return {"neutral": 1.0} # Default
        return self._current_result

class MockVoiceEmotionDetector(EmotionDetector):
    """
    Mock Voice Emotion Detector.
    Allows manual setting of the result for testing purposes.
    """
    def __init__(self):
        self._current_result: Optional[Dict[str, float]] = None

    def set_result(self, result: Dict[str, float]):
        """Sets the result to be returned by detect()."""
        self._current_result = result

    def detect(self) -> Dict[str, float]:
        if self._current_result is None:
            return {"neutral": 1.0} # Default
        return self._current_result
