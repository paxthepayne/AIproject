import datetime
import pandas as pd
from difflib import get_close_matches
import streamlit as st
import os

# ---- Load CSV ----
@st.cache_data
def load_CL():
    base_dir = os.path.dirname(__file__)
    CL_path = os.path.join(base_dir, "CL.csv")
    return pd.read_csv(CL_path)

CL = load_CL()

if CL.empty:
    st.stop()

# ---- MODULES ----
class DataManager:
    def get_current_context(self, weather_choice):
        return {
            "time": datetime.datetime.now(),
            "weather": weather_choice,
            "events": []
        }

class CrowdPredictor:
    def predict_crowd(self, location, context):
        names = CL['name'].tolist()
        closest = get_close_matches(location, names, n=1, cutoff=0.6)

        if not closest:
            return None, "⚠️ No close match found!"

        best_match = closest[0]
        weather = context['weather']

        row = CL[CL['name'] == best_match].iloc[0]
        base_value = row.get("baseline_crowd_levels", row.get("crowd_index", None))

        if base_value is None:
            return None, "❌ No crowd level column found!"

        modifier = 2 if weather == "Sunny" else (0.5 if weather == "Rainy" else 1)
        estimated = min(100, base_value * modifier)

        return estimated, f"Closest match: **{best_match}**"

class SmartRecommender:
    def __init__(self, predictor):
        self.predictor = predictor

    def get_alternatives(self, desired_destination, context):
        alternatives = ["Park de la Ciutadella", "Hospital de Sant Pau"]
        rankings = []

        for alt in alternatives:
            lvl, _ = self.predictor.predict_crowd(alt, context)
            if lvl is not None:
                rankings.append((alt, lvl))

        return sorted(rankings, key=lambda x: x[1])

# ---- STREAMLIT UI ----
st.title("🌆 Barcelona Smart Tourist Guide")
st.write("Reroute tourists to avoid crowded areas and ease city congestion")

location_input = st.text_input("📍 Enter a destination:", "Sagrada Família - Basilica")
weather_choice = st.selectbox("☁️ Select weather:", ["Sunny", "Cloudy", "Rainy"])

if st.button("Predict Crowd Level"):
    st.subheader("🔍 Results")

    dm = DataManager()
    predictor = CrowdPredictor()
    context = dm.get_current_context(weather_choice)

    crowd_level, info_msg = predictor.predict_crowd(location_input, context)

    if info_msg:
        st.info(info_msg)

    if crowd_level is not None:
        st.metric(label="Estimated Crowd Level", value=f"{int(crowd_level)}/100")

        CROWD_THRESHOLD = 70
        if crowd_level > CROWD_THRESHOLD:
            st.warning("High congestion expected!")

            recommender = SmartRecommender(predictor)
            alternatives = recommender.get_alternatives(location_input, context)

            if alternatives:
                st.write("### 🏖 Recommended Alternatives")
                for name, lvl in alternatives:
                    st.write(f"- **{name}** → {int(lvl)}/100")
    else:
        st.error("Could not estimate crowd level.")

st.caption("🔧 Prototype model – planned upgrades: hourly predictions, weather API, events API, smart routing")
