import os
import sys

# === Make sure Python can find the Model folder ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CURRENT_DIR, "Model")

if MODEL_DIR not in sys.path:
    sys.path.append(MODEL_DIR)

from stage_model import StageDurationModel

# === Create model wrapper and load the saved pipeline ===
model = StageDurationModel("Model/stage_duration_model.joblib")
model.load()

# === Example input to test prediction ===
input_data = {
    "issue_type": "Feature",
    "priority": "High",
    "team": "Team A",
    "current_stage": "Testing",
    "story_points": 8,
    "previous_stages_count": 1,
    "num_dependencies": 2
}

# === Make prediction ===
prediction = model.predict(input_data)
print("Predicted days in stage:", prediction)

