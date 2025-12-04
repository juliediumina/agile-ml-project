import os
import sys
from fastapi import FastAPI
from pydantic import BaseModel

# === Make sure Python can find the Model folder ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))       # .../agile-ml-project/API
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)                    # .../agile-ml-project
MODEL_DIR = os.path.join(PROJECT_ROOT, "Model")                # .../agile-ml-project/Model

if MODEL_DIR not in sys.path:
    sys.path.append(MODEL_DIR)

# Import the wrapper class from stage_model.py (same folder as .joblib)
from stage_model import StageDurationModel



# === Pydantic schema for request body ===
class StageInput(BaseModel):
    issue_type: str
    priority: str
    team: str
    current_stage: str
    story_points: int
    previous_stages_count: int
    num_dependencies: int


# === Initialize FastAPI app and load model ===
app = FastAPI(
    title="Agile Stage Duration Predictor",
    description="Predicts how many days a task will stay in the current stage based on agile ticket features.",
    version="1.0.0",
)

model = StageDurationModel(os.path.join(MODEL_DIR, "stage_duration_model.joblib"))
model.load()


# === Simple health check ===
@app.get("/")
def read_root():
    return {"message": "Agile Stage Duration API is running 🎯"}


# === Prediction endpoint ===
@app.post("/predict")
def predict_duration(data: StageInput):
    # Convert Pydantic model to dict
    input_dict = data.dict()
    prediction = model.predict(input_dict)
    return {
        "input": input_dict,
        "predicted_days_in_stage": float(prediction),
    }

