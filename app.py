import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Property Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("xgb_model.pkl")   # make sure this file is in the same folder

model = load_model()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🏠 Property Price Predictor")
st.markdown("Enter the property details below to get an estimated sale price.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    living_area = st.number_input(
        "Living Area (sq ft)",
        min_value=100, max_value=20000,
        value=1800, step=50
    )
    bedrooms = st.number_input(
        "Bedrooms",
        min_value=0, max_value=20,
        value=3, step=1
    )

with col2:
    bathrooms = st.number_input(
        "Bathrooms",
        min_value=0.0, max_value=20.0,
        value=2.0, step=0.5
    )
    lot_size = st.number_input(
        "Lot Size (sq ft)",
        min_value=0, max_value=500000,
        value=6000, step=100
    )

st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("Predict Price", type="primary", use_container_width=True):

    # Build a single-row DataFrame matching what the model was trained on.
    # The model expects ALL original feature columns — we fill the rest with 0
    # and overwrite the four fields the user provided.
    try:
        # Get the exact feature list the model was trained on
        feature_names = model.get_booster().feature_names

        input_df = pd.DataFrame(
            np.zeros((1, len(feature_names))),
            columns=feature_names
        )

        # Map user inputs to whatever column names your training data used.
        # Adjust these keys if your CSV used different names.
        name_map = {
            "LivingArea":      living_area,
            "BedroomsTotal":   bedrooms,
            "BathroomsTotalInteger": bathrooms,
            "LotSizeSquareFeet": lot_size,
        }

        for col, val in name_map.items():
            if col in input_df.columns:
                input_df[col] = val
            else:
                st.warning(f"Column '{col}' not found in model features — skipped.")

        prediction = model.predict(input_df)[0]

        st.success(f"### Estimated Sale Price: **${prediction:,.0f}**")

        # Rough confidence band based on your model's ~13% MAPE
        low  = prediction * 0.87
        high = prediction * 1.13
        st.caption(f"Typical range (±13% MAPE): ${low:,.0f} – ${high:,.0f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("Make sure `xgb_model.pkl` is in the same folder as `app.py`.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Model: XGBoost  •  Trained on MLS property data  •  R² = 0.89")
