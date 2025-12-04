import os
import sys

import streamlit as st

# === Make sure Python can find the Model folder ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CURRENT_DIR, "Model")

if MODEL_DIR not in sys.path:
    sys.path.append(MODEL_DIR)

from stage_model import StageDurationModel


# === Load model once ===
@st.cache_resource
def load_model():
    model_path = os.path.join(MODEL_DIR, "stage_duration_model.joblib")
    m = StageDurationModel(model_path)
    m.load()
    return m


model = load_model()

# === UI ===
st.set_page_config(page_title="Agile Stage Duration Predictor", page_icon="⏱️")

st.title("⏱️ Agile Stage Duration Predictor")
#description
st.write("""
This tool predicts how many days a task is likely to remain in its current Agile stage,
based on factors like story points, priority, team, dependencies, and workflow history.

The goal is to help teams identify potential delays early and improve sprint planning,
risk management, and workflow transparency.
""")
#explanation of risk level
st.markdown("""
### 📊 How to interpret risk levels

- **🟢 Low risk** — The task is expected to move quickly through the current stage.  
- **🟠 Medium risk** — The task may require additional attention or follow-up.  
- **🔴 High risk** — The task is likely to cause delays and may block the sprint if not addressed early.

Risk level is based on the predicted duration compared to typical patterns for similar tasks.
""")
# --- Feature explanations ---
with st.expander("ℹ️ What do these inputs mean?"):
    st.markdown(
        """
- **Issue type** – The kind of work item (e.g., *Bug*, *Feature*, *Task*).  
  Different issue types often have different cycle times.

- **Priority** – How urgent or important the task is (*Low, Medium, High*).  
  Higher priority items are usually expected to move faster.

- **Team** – Which Agile team owns the task (Team A / Team B / Team C, etc.).  
  Each team can have its own workflow and speed.

- **Current stage** – The column where the task is right now  
  (e.g., *To Do*, *In Progress*, *Code Review*, *Testing*).  
  The model predicts how long it will remain in **this** stage.

- **Story points** – A relative estimate of **effort + complexity + uncertainty**,  
  not hours. A task with 8 points should feel “bigger” than one with 3 points.

- **Previous stages completed** – How many stages the task has already passed  
  (for example: from *Backlog → To Do → In Progress* = 2 stages completed).

- **Number of dependencies** – How many other tasks, approvals, or systems this task depends on.  
  More dependencies usually means more chances for delays.
        """
    )

st.markdown("---")
st.write("Estimate how many days a task will stay in its current Agile stage.")

issue_types = ["Bug", "Feature", "Improvement"]
priorities = ["Low", "Medium", "High", "Critical"]
teams = ["Team A", "Team B", "Team C"]
stages = ["Backlog", "In Progress", "Code Review", "Testing", "Done"]

col1, col2 = st.columns(2)

with col1:
    issue_type = st.selectbox("Issue type", issue_types)
    priority = st.selectbox("Priority", priorities)
    team = st.selectbox("Team", teams)
    current_stage = st.selectbox("Current stage", stages)

with col2:
    story_points = st.slider("Story points", 1, 21, 5)
    previous_stages_count = st.slider("Previous stages completed", 0, 6, 1)
    num_dependencies = st.slider("Number of dependencies", 0, 10, 2)

st.markdown("---")

if st.button("🔮 Predict duration"):
    input_data = {
        "issue_type": issue_type,
        "priority": priority,
        "team": team,
        "current_stage": current_stage,
        "story_points": story_points,
        "previous_stages_count": previous_stages_count,
        "num_dependencies": num_dependencies,
    }

    prediction = float(model.predict(input_data))
    days = round(prediction, 2)

    if days <= 2:
        risk_label = "Low risk"
        color = "green"
        emoji = "🟢"
    elif days <= 7:
        risk_label = "Medium risk"
        color = "orange"
        emoji = "🟠"
    else:
        risk_label = "High risk"
        color = "red"
        emoji = "🔴"

    st.subheader("Result")
    st.markdown(
        f"<h2 style='text-align:center;'>Estimated: <b>{days} days</b></h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<h4 style='text-align:center; color:{color};'>{emoji} {risk_label}</h4>",
        unsafe_allow_html=True,
    )

    with st.expander("See input details"):
        st.json(input_data)
else:
    st.info("Set ticket parameters and click Predict.")
