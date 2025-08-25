import streamlit as st
import pandas as pd
import joblib

# Load saved model, scaler, and expected columns
model = joblib.load(r"C:\Users\ysanc\OneDrive\Desktop\Callorie_Intake_calculator\notebook\R_F_Regressor_heart.pkl")
scaler = joblib.load(r"C:\Users\ysanc\OneDrive\Desktop\Callorie_Intake_calculator\notebook\preprocessor.pkl")
expected_columns = joblib.load(r"C:\Users\ysanc\OneDrive\Desktop\Callorie_Intake_calculator\notebook\columns.pkl")

st.title("Daily calorie intake calculator")
st.markdown("Provide the following details to know callorie you needs:")

# Collect user input
Age = st.slider("Age", 5, 100, 40)
Gender = st.selectbox("Gender", ["Male", "Female"])
Working_Type = st.selectbox("Working Type", ['Unemployed', 'Desk Job', 'Freelancer', 'Healthcare', 'Retired', 'Manual Labor', 'Student', 'Self-Employed'])
Sleep_Hours = st.number_input("Sleep Hours", 0, 24, 8)
Height_m = st.number_input("Height (in m)", 0.25, 1.95, 1.65)


# When Predict is clicked
if st.button("Predict"):

    # Create a raw input dictionary
    raw_input = {
        'Age': Age,
        'Gender': Gender,
        'Working_Type': Working_Type,
        'Sleep_Hours': Sleep_Hours,
        'Height_m': Height_m
    }

    # Create input dataframe
    input_df = pd.DataFrame([raw_input])

    # Fill in missing columns with 0s
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns
    input_df = input_df[expected_columns]

    # Scale the input
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(scaled_input)

    # Display result
    st.success(f"Estimated daily calorie intake: {prediction[0]:.2f} kcal")

