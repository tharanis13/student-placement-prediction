# app.py

import streamlit as st
import pandas as pd
import joblib
import PyPDF2

# Load model, scaler, and target label encoder
model = joblib.load(r"C:\mlproject\placement_model.pkl")
scaler = joblib.load(r"C:\mlproject\scaler.pkl")
target_le = joblib.load(r"C:\mlproject\target_le.pkl")

st.set_page_config(page_title="Student Placement Predictor", layout="wide")

# Display home.html
with open("home.html", "r", encoding="utf-8") as f:
    html_content = f.read()
st.components.v1.html(html_content, height=300)

st.markdown("### Upload your resume (PDF) and/or enter your details:")

# Resume upload
resume_file = st.file_uploader("Upload Resume", type=["pdf"])
resume_text = ""
if resume_file:
    pdf_reader = PyPDF2.PdfReader(resume_file)
    for page in pdf_reader.pages:
        resume_text += page.extract_text() + "\n"
    st.text_area("Resume Preview", resume_text, height=200)

# Student info form
st.markdown("### Enter Student Details:")
cgpa = st.number_input("CGPA", 0.0, 10.0, 6.5, 0.01)
ssc_marks = st.number_input("SSC Marks (%)", 0, 100, 75)
hsc_marks = st.number_input("HSC Marks (%)", 0, 100, 75)
aptitude_score = st.number_input("Aptitude Test Score", 0, 100, 50)

internships = st.selectbox("Internships", ["No", "Yes"])
projects = st.selectbox("Projects", ["Few", "Many"])
workshops = st.selectbox("Workshops/Certifications", ["Low", "Medium", "High"])
softskills = st.selectbox("Soft Skills Rating", ["Low", "Medium", "High"])
extracurricular = st.selectbox("Extracurricular Activities", ["Low", "Medium", "High"])
placement_training = st.selectbox("Placement Training", ["No", "Yes"])

# Predict button
if st.button("Predict Placement Status"):

    # Create input dataframe
    input_df = pd.DataFrame({
        "CGPA": [cgpa],
        "SSC_Marks": [ssc_marks],
        "HSC_Marks": [hsc_marks],
        "AptitudeTestScore": [aptitude_score],
        "SoftSkillsRating": [softskills],
        "ExtracurricularActivities": [extracurricular],
        "Projects": [projects],
        "Workshops/Certifications": [workshops],
        "Internships": [internships],
        "PlacementTraining": [placement_training]
    })

    # Map categorical variables
    cat_map = {
        "Internships": {"No":0,"Yes":1},
        "Projects":{"Few":0,"Many":1},
        "Workshops/Certifications":{"Low":0,"Medium":1,"High":2},
        "SoftSkillsRating":{"Low":0,"Medium":1,"High":2},
        "ExtracurricularActivities":{"Low":0,"Medium":1,"High":2},
        "PlacementTraining":{"No":0,"Yes":1}
    }

    for col in cat_map:
        input_df[col] = input_df[col].map(cat_map[col])

    # Ensure correct column order
    feature_cols = [
        "CGPA",
        "Internships",
        "Projects",
        "Workshops/Certifications",
        "AptitudeTestScore",
        "SoftSkillsRating",
        "ExtracurricularActivities",
        "PlacementTraining",
        "SSC_Marks",
        "HSC_Marks"
    ]
    input_df = input_df[feature_cols]

    # Scale and predict
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)[0][1]*100
    result = target_le.inverse_transform(prediction)[0]

    # Display results
    if result.lower() == "placed":
        st.success(f"✅ Prediction: {result}")
        st.progress(int(prediction_proba))
    else:
        st.error(f"❌ Prediction: {result}")
        st.progress(int(prediction_proba))

    st.info(f"Probability of being placed: {prediction_proba:.2f}%")

    st.markdown("---")
    st.subheader("Entered Details Summary")
    st.table(input_df.T.rename(columns={0:"Value"}))
