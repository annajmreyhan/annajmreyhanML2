
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

# Load model dan tools
model = tf.lite.Interpreter(model_path="model.tflite")
model.allocate_tensors()
input_details = model.get_input_details()
output_details = model.get_output_details()

scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoder.pkl")

# Judul Aplikasi
st.title("Prediksi Kelulusan Siswa (Math Score >= 70)")

# Input user
gender = st.selectbox("Gender", label_encoders['gender'].classes_)
ethnicity = st.selectbox("Race/Ethnicity", label_encoders['race/ethnicity'].classes_)
education = st.selectbox("Parental Level of Education", label_encoders['parental level of education'].classes_)
lunch = st.selectbox("Lunch", label_encoders['lunch'].classes_)
prep = st.selectbox("Test Preparation Course", label_encoders['test preparation course'].classes_)
reading = st.slider("Reading Score", 0, 100, 70)
writing = st.slider("Writing Score", 0, 100, 70)

if st.button("Prediksi"):
    # Encode
    data = {
        'gender': [label_encoders['gender'].transform([gender])[0]],
        'race/ethnicity': [label_encoders['race/ethnicity'].transform([ethnicity])[0]],
        'parental level of education': [label_encoders['parental level of education'].transform([education])[0]],
        'lunch': [label_encoders['lunch'].transform([lunch])[0]],
        'test preparation course': [label_encoders['test preparation course'].transform([prep])[0]],
        'reading score': [reading],
        'writing score': [writing]
    }
    df_input = pd.DataFrame(data)
    scaled_input = scaler.transform(df_input)

    # TFLite prediction
    model.set_tensor(input_details[0]['index'], np.array(scaled_input, dtype=np.float32))
    model.invoke()
    output = model.get_tensor(output_details[0]['index'])
    prediction = (output > 0.5).astype(int)[0][0]

    if prediction == 1:
        st.success("Prediksi: LULUS (Math Score >= 70)")
    else:
        st.warning("Prediksi: TIDAK LULUS (Math Score < 70)")
