import streamlit as st
import pandas as pd
import joblib

# Load saved model, scaler, and expected columns
model = joblib.load(r"C:\Users\ysanc\OneDrive\Desktop\Callorie_Intake_calculator\notebook\R_F_Regressor_heart.pkl")
scaler = joblib.load(r"C:\Users\ysanc\OneDrive\Desktop\Callorie_Intake_calculator\notebook\preprocessor.pkl")
expected_columns = joblib.load(r"C:\Users\ysanc\OneDrive\Desktop\Callorie_Intake_calculator\notebook\columns.pkl")

# Title and description
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🥗 Daily Calorie Intake Calculator</h1>", unsafe_allow_html=True)
st.markdown("### Provide your details to estimate your daily calorie needs.")

# Layout with columns
col1, col2 = st.columns(2)
with col1:
    Age = st.slider("📅 Age", 5, 100, 25)
    Sleep_Hours = st.number_input("😴 Sleep Hours", 0, 24, 7)

with col2:
    Gender = st.selectbox("⚧️ Gender", ["Male", "Female"])
    Working_Type = st.selectbox("💼 Working Type", ['Unemployed', 'Desk Job', 'Freelancer', 'Healthcare', 'Retired', 'Manual Labor', 'Student', 'Self-Employed'])
    Height_m = st.number_input("📏 Height (in meters)", 0.25, 2.5, 1.68)


# Prediction
if st.button("🔍 Predict"):
    raw_input = {
        'Age': Age,
        'Gender': Gender,
        'Working_Type': Working_Type,
        'Sleep_Hours': Sleep_Hours,
        'Height_m': Height_m
    }

    input_df = pd.DataFrame([raw_input])

    # Fill missing columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)

    
# Display result
    st.success(f"🔥 Estimated Daily Calorie Intake: **{prediction[0]:.2f} kcal**")


# Info section
with st.expander("ℹ️ Why these inputs matter?"):
    st.write("""
    - **Age** and **Gender** influence your basal metabolic rate.
    - **Working Type** reflects your daily activity level.
    - **Sleep Hours** affect metabolism and recovery.
    - **Height** helps estimate your body composition.
    """)

# Custom button style
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 0.5em 2em;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)




