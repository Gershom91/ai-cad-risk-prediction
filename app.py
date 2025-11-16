import streamlit as st
import numpy as np
import pickle

# --------------------
# PAGE CONFIG
# --------------------
st.set_page_config(
    page_title="CAD Risk Prediction",
    page_icon="💗",
    layout="centered",
)


# --------------------
# LOAD MODEL + SCALER
# --------------------
with open("cad_random_forest_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


# --------------------
# CUSTOM CSS (Styling)
# --------------------
st.markdown(
    """
    <style>

    /* Background gradient */
    .stApp {
        background: linear-gradient(135deg, #0f0f0f, #1a1a1a);
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Make widgets look premium */
    .stTextInput>div>div>input,
    .stNumberInput>div>input,
    .stSelectbox>div>div {
        background-color: #222222;
        color: white;
        border: 1px solid #5a5a5a;
        border-radius: 6px;
    }

    /* Card-style container */
    .card {
        background-color: #161616;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #333333;
        margin-bottom: 20px;
    }

    /* Title styling */
    .title {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        padding-bottom: 10px;
    }

    /* Risk result styling */
    .risk-high {
        background-color: #8B0000;
        padding: 15px;
        border-radius: 10px;
        color: white;
        font-size: 20px;
        text-align: center;
    }

    .risk-low {
        background-color: #004d1a;
        padding: 15px;
        border-radius: 10px;
        color: white;
        font-size: 20px;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------
# SIDEBAR DETAILS
# --------------------
st.sidebar.title("📁 Project Info")
st.sidebar.markdown(
    """
    **AI CAD Risk Predictor v1.0**

    - Machine Learning model: **Random Forest**
    - Optimized for binary CAD prediction  
    - Based on clinical diagnostic features  
    - For educational/medical research purposes  
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ by Gershom")


# --------------------
# MAIN TITLE
# --------------------
st.markdown('<div class="title">💗 CAD Risk Prediction App</div>', unsafe_allow_html=True)

st.write("Enter the patient diagnostic data below to estimate the probability of Coronary Artery Disease (CAD).")


# --------------------
# CARD CONTAINER
# --------------------
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    age = st.number_input("Age (years)", 20, 90, 55)
    sex = st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 130)
    chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 240)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)", [1, 0])
    restecg = st.selectbox("Resting ECG (0, 1, 2)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise-Induced Angina (1=True, 0=False)", [1, 0])
    oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 6.5, 1.0)
    slope = st.selectbox("Slope (0, 1, 2)", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels Colored (0–3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia (1, 2, 3)", [1, 2, 3])

    st.markdown('</div>', unsafe_allow_html=True)


# --------------------
# PREDICTION PIPELINE
# --------------------
input_data = np.array(
    [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
).reshape(1, -1)

numeric_indices = [0, 3, 4, 7, 9]
input_data[:, numeric_indices] = scaler.transform(input_data[:, numeric_indices])

if st.button("🔍 Predict CAD Risk"):
    pred = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    if pred == 1:
        st.markdown(
            f'<div class="risk-high">⚠️ HIGH RISK of CAD<br>Estimated Probability: {proba:.2%}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="risk-low">✅ LOW RISK of CAD<br>Estimated Probability: {proba:.2%}</div>',
            unsafe_allow_html=True,
        )


# --------------------
# PROFESSIONAL DISCLAIMER
# --------------------
st.markdown(
    """
    <hr>
    <small>
    This AI tool is intended for educational and research purposes only.  
    It is **not** a substitute for professional medical diagnosis, evaluation, or treatment.  
    Always consult a licensed physician for clinical decisions.
    </small>
    """,
    unsafe_allow_html=True,
)
