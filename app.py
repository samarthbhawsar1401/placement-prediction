import streamlit as st
import pickle
import pandas as pd

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


st.set_page_config(
    page_title="Placement Predictor",
    page_icon="🎓"
)

st.title("🎓 Placement Prediction")
st.write("Enter your CGPA and IQ to predict placement.")


cgpa = st.number_input(
    "CGPA",
    min_value=0.0,
    max_value=10.0,
    value=7.0,
    step=0.1
)

iq = st.number_input(
    "IQ",
    min_value=50,
    max_value=200,
    value=100,
    step=1
)


if st.button("Predict Placement"):

    input_data = pd.DataFrame(
        [[cgpa, iq]],
        columns=["cgpa", "iq"]
    )

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.success("🎉 Prediction: You are likely to be placed!")
    else:
        st.error("😔 Prediction: You may not be placed.")
