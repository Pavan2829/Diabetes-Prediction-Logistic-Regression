import streamlit as st
import joblib
import numpy as np

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🩺 Diabetes Prediction")

st.sidebar.markdown("---")

st.sidebar.subheader("Model Information")

st.sidebar.write("**Algorithm:** Logistic Regression")
st.sidebar.write("**Dataset:** Pima Indians Diabetes")
st.sidebar.write("**Accuracy:** 75.32%")
st.sidebar.write("**Cross Validation:** 77.04%")

st.sidebar.markdown("---")

st.sidebar.info(
    "Enter the patient's medical information and click "
    "**Predict Diabetes** to estimate the risk."
)

st.sidebar.markdown("---")

st.sidebar.success("Developed by Pavan Jangala")

# -----------------------------
# Main Title
# -----------------------------
st.title("🩺 Diabetes Prediction System")

st.caption(
    "Machine Learning Web Application using Logistic Regression"
)

st.markdown("---")

# -----------------------------
# Input Section
# -----------------------------
st.header("Patient Information")

col1, col2 = st.columns(2)

with col1:

    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        value=1
    )

    glucose = st.number_input(
        "Glucose",
        min_value=0,
        value=120
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0,
        value=70
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0,
        value=20
    )

with col2:

    insulin = st.number_input(
        "Insulin",
        min_value=0,
        value=80
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        value=25.0
    )

    dpf = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        value=0.50
    )

    age = st.number_input(
        "Age",
        min_value=1,
        value=30
    )

st.markdown("")

predict = st.button(
    "🔍 Predict Diabetes",
    use_container_width=True
)

# -----------------------------
# Prediction
# -----------------------------
if predict:

    input_data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    probability = model.predict_proba(input_scaled)

    diabetic_probability = probability[0][1]
    non_diabetic_probability = probability[0][0]

    st.markdown("---")

    if prediction[0] == 1:
        st.error("⚠️ Prediction: Diabetic")
    else:
        st.success("✅ Prediction: Non-Diabetic")

    st.subheader("Prediction Confidence")

    st.progress(float(diabetic_probability))

    st.write(
        f"**Diabetic Probability:** {diabetic_probability*100:.2f}%"
    )

    st.write(
        f"**Non-Diabetic Probability:** {non_diabetic_probability*100:.2f}%"
    )

    st.markdown("---")

    st.subheader("Patient Summary")

    summary1, summary2 = st.columns(2)

    with summary1:

        st.write(f"**Pregnancies:** {pregnancies}")
        st.write(f"**Glucose:** {glucose}")
        st.write(f"**Blood Pressure:** {blood_pressure}")
        st.write(f"**Skin Thickness:** {skin_thickness}")

    with summary2:

        st.write(f"**Insulin:** {insulin}")
        st.write(f"**BMI:** {bmi}")
        st.write(f"**DPF:** {dpf}")
        st.write(f"**Age:** {age}")

# -----------------------------
# Model Metrics
# -----------------------------
st.markdown("---")

st.subheader("Model Performance")

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric("Accuracy", "75.32%")
metric2.metric("Precision", "67%")
metric3.metric("Recall", "62%")
metric4.metric("F1 Score", "64%")

# -----------------------------
# About Project
# -----------------------------
st.markdown("---")

st.subheader("About Project")

st.write("""
This web application predicts the likelihood of diabetes using a
Logistic Regression machine learning model.

### Workflow

- Data Cleaning
- Missing Value Handling
- Feature Scaling
- Logistic Regression
- Hyperparameter Tuning
- Model Evaluation
- Streamlit Deployment

Dataset Used:
Pima Indians Diabetes Dataset
""")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")

st.caption("© 2026 Pavan Kumar Jangala | Machine Learning Portfolio Project")