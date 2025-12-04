import joblib     # For loading the saved model
import pandas as pd      

class StageDurationModel:
    def __init__(self, model_path="stage_duration_model.joblib"):
        self.model_path = model_path
        self.model = None

    def load(self):
        """Load trained model from file."""
        self.model = joblib.load(self.model_path)

    def predict(self, input_dict):
        """Predict using a dictionary input."""
        df = pd.DataFrame([input_dict])
        return self.model.predict(df)[0]

