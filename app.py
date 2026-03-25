import streamlit as st
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="EasyBP | 10-Year CHD Risk Predictor",
    page_icon="🫀",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
    <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: #C0392B;
        }
        .subtitle {
            font-size: 1rem;
            color: #555;
            margin-bottom: 1.5rem;
        }
        .risk-box-high {
            background-color: #FDEDEC;
            border-left: 5px solid #E74C3C;
            padding: 15px 20px;
            border-radius: 8px;
            margin-top: 10px;
        }
        .risk-box-low {
            background-color: #EAFAF1;
            border-left: 5px solid #2ECC71;
            padding: 15px 20px;
            border-radius: 8px;
            margin-top: 10px;
        }
        .disclaimer {
            font-size: 0.8rem;
            color: #888;
            margin-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)


# ─── Load Model & Scaler ──────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    repo_root = Path(__file__).resolve().parent
    model_path = repo_root / "best_model.pkl"
    scaler_path = repo_root / "scaler.pkl"

    if not model_path.exists() or not scaler_path.exists():
        missing = []
        if not model_path.exists():
            missing.append(str(model_path))
        if not scaler_path.exists():
            missing.append(str(scaler_path))
        raise FileNotFoundError("Missing artifact(s): " + ", ".join(missing))

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_artifacts()
    model_loaded = True
except FileNotFoundError as e:
    model_loaded = False
    st.error("❌ Could not load model/scaler. Ensure `best_model.pkl` and "
             "`scaler.pkl` are in the same directory.")
except ModuleNotFoundError as e:
    model_loaded = False
    st.error("❌ Could not unpickle model/scaler due to missing package. "
             "Install scikit-learn in the active environment.")
except Exception as e:
    model_loaded = False
    st.error("❌ Unexpected error loading artifacts.")


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/heart-with-pulse.png", width=80)
    st.markdown("## 🫀 EasyBP")
    st.markdown("""
    **EasyBP** predicts your **10-year risk** of developing 
    **Coronary Heart Disease (CHD)** using key clinical biomarkers.

    ---
    **Model**: Random Forest Classifier  
    **Trained on**: Framingham Heart Study Dataset  
    **Features used**: 7 clinical variables  

    ---
    > ⚕️ *For educational use only. Always consult a medical professional.*
    """)


# ─── Main App ─────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🫀 EasyBP — CHD Risk Predictor</p>', unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Estimate a patient's 10-year risk of Coronary Heart Disease using clinical biomarkers.</p>", unsafe_allow_html=True)

st.divider()

# ─── Input Form ───────────────────────────────────────────────────────────────
st.subheader("🩺 Enter Patient Clinical Data")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "🎂 Age (years)",
        min_value=20, max_value=100, value=45, step=1,
        help="Patient's age in years."
    )
    bmi = st.number_input(
        "⚖️ BMI (kg/m²)",
        min_value=10.0, max_value=70.0, value=25.0, step=0.1,
        help="Body Mass Index."
    )
    sys_bp = st.number_input(
        "📈 Systolic Blood Pressure (mmHg)",
        min_value=80.0, max_value=300.0, value=120.0, step=0.5,
        help="Systolic blood pressure measurement."
    )
    dia_bp = st.number_input(
        "📉 Diastolic Blood Pressure (mmHg)",
        min_value=40.0, max_value=150.0, value=80.0, step=0.5,
        help="Diastolic blood pressure measurement."
    )

with col2:
    heart_rate = st.number_input(
        "💓 Heart Rate (bpm)",
        min_value=30.0, max_value=200.0, value=75.0, step=1.0,
        help="Resting heart rate in beats per minute."
    )
    tot_chol = st.number_input(
        "🧪 Total Cholesterol (mg/dL)",
        min_value=100.0, max_value=700.0, value=220.0, step=1.0,
        help="Total cholesterol level."
    )
    glucose = st.number_input(
        "🍬 Glucose Level (mg/dL)",
        min_value=40.0, max_value=400.0, value=80.0, step=1.0,
        help="Fasting blood glucose level."
    )

st.divider()

# ─── Prediction ───────────────────────────────────────────────────────────────
predict_btn = st.button("🔍 Predict 10-Year CHD Risk", use_container_width=True, type="primary")

if predict_btn:
    if not model_loaded:
        st.warning("⚠️ Model artifacts are not loaded. Please check your pickle files.")
    else:
        # Assemble input in correct feature order
        feature_names = ["age", "BMI", "sysBP", "diaBP", "heartRate", "totChol", "glucose"]
        input_values = [age, bmi, sys_bp, dia_bp, heart_rate, tot_chol, glucose]
        input_df = pd.DataFrame([input_values], columns=feature_names)

        # Scale features
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)[0]

        try:
            prob = model.predict_proba(input_scaled)[0]
            if len(prob) > 1:
                risk_pct = prob[1] * 100
                no_risk_pct = prob[0] * 100
            else:
                risk_pct = 100.0 if prediction == 1 else 0.0
                no_risk_pct = 0.0 if prediction == 1 else 100.0
        except (AttributeError, ValueError):
            # Some estimators don't support predict_proba
            risk_pct = 100.0 if prediction == 1 else 0.0
            no_risk_pct = 0.0 if prediction == 1 else 100.0

        # ── Result Display ────────────────────────────────────────────────
        st.subheader("📊 Prediction Result")

        if prediction == 1:
            st.markdown(f"""
            <div class="risk-box-high">
                <h3 style="color:#C0392B; margin:0">⚠️ HIGH RISK</h3>
                <p style="margin:5px 0 0 0; color: black;">This patient is at <strong>high risk</strong> of developing 
                Coronary Heart Disease within the next 10 years.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-box-low">
                <h3 style="color:#1E8449; margin:0">✅ LOW RISK</h3>
                <p style="margin:5px 0 0 0; color: black;">This patient is at <strong>low risk</strong> of developing 
                Coronary Heart Disease within the next 10 years.</p>
            </div>
            """, unsafe_allow_html=True)

        # ── Probability Metrics ───────────────────────────────────────────
        st.markdown("")
        m1, m2, m3 = st.columns(3)
        m1.metric("🔴 CHD Risk Probability", f"{risk_pct:.2f}%")
        m2.metric("🟢 No-Risk Probability", f"{no_risk_pct:.2f}%")
        m3.metric("🏷️ Prediction", "At Risk" if prediction == 1 else "No Risk")

        # ── Risk Gauge Bar ────────────────────────────────────────────────
        st.markdown("**Risk Level Gauge**")
        risk_pct_clamped = max(0, min(100, int(round(risk_pct))))
        st.progress(risk_pct_clamped)
        if risk_pct_clamped < 20:
            st.caption("🟢 Low Risk Zone")
        elif risk_pct < 50:
            st.caption("🟡 Moderate Risk Zone")
        else:
            st.caption("🔴 High Risk Zone")

        # ── Input Summary Table ───────────────────────────────────────────
        st.divider()
        st.subheader("📋 Patient Data Summary")
        summary_df = pd.DataFrame({
            "Feature": ["Age", "BMI", "Systolic BP", "Diastolic BP", "Heart Rate", "Total Cholesterol", "Glucose"],
            "Value": [age, bmi, sys_bp, dia_bp, heart_rate, tot_chol, glucose],
            "Unit": ["years", "kg/m²", "mmHg", "mmHg", "bpm", "mg/dL", "mg/dL"],
            "Reference Range": [
                "32–70", "18.5–24.9", "90–120", "60–80",
                "60–100", "< 200 (desirable)", "70–99 (fasting)"
            ]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ─── Footer Disclaimer ────────────────────────────────────────────────────────
st.markdown("""
<p class="disclaimer">
⚕️ <strong>Medical Disclaimer:</strong> This tool is developed for educational and research purposes only. 
It is not intended to replace professional medical diagnosis, advice, or treatment. 
Always consult a qualified healthcare provider for clinical decisions.
</p>
""", unsafe_allow_html=True)

