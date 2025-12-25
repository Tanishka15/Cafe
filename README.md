# Multimodal Emotion Fusion Framework

A robust framework for detecting emotions by fusing signals from Face and Voice modalities. This system implements a dynamic weighting mechanism that intelligently prioritizes auditory cues when facial expressions are unreliable or ambiguous.

## Features

- **Multimodal Fusion**: Combines probabilities from Face and Voice detectors.
- **Dynamic Uncertainty Handling**:
  - Automatically lowers reliance on facial data if confidence drops below a threshold (0.6).
  - Prioritizes voice data in low-visibility or ambiguous facial scenarios.
- **Mock Detectors**: Includes extensible `MockFaceEmotionDetector` and `MockVoiceEmotionDetector` for testing scenarios.
- **Scenario Runner**: Built-in test scenarios (Reliable, Unreliable, Ambiguous Edge Case) to verify logic.

## Project Structure

```bash
.
├── main.py             # Entry point: Runs defined scenarios
├── detectors.py        # Abstract base class and Mock implementation of detectors
├── fusion_engine.py    # Core logic: Uncertainty rules and weighted fusion
└── __pycache__/
```

## Setup & Usage

### 1. Requirements
- Python 3.7+
- No external heavy dependencies required for the core logic (uses standard libraries).

### 2. Run the Application
Execute the main script to see the fusion logic in action across pre-defined scenarios:

```bash
python3 main.py
```

### 3. Expected Output
The system prints decision logs showing:
- Raw input probabilities.
- Calculated confidence scores.
- Weights applied (`face` vs `voice`).
- Reasoning for the weighting decision (e.g., "Prioritizing Voice").
- Final fused emotion probabilities.

## Configuration
You can tune the fusion thresholds in `fusion_engine.py`:
- `self.default_face_weight`: Standard weight when face is reliable (Default: 0.6).
- `self.face_confidence_threshold`: Confidence cutoff (Default: 0.6).
