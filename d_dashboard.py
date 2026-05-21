import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
import datetime
import os
from fpdf import FPDF
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai

st.set_page_config(page_title="MediPredict AI", page_icon="🏥", layout="wide")

# ---------------- GEMINI API ----------------
import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

chat_model = genai.GenerativeModel("gemini-2.5-flash")

# ---------------- UI CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eef7ff 0%, #f8fafc 50%, #ecfeff 100%);
}
.block-container {
    padding-top: 1rem;
}
.hero {
    background: linear-gradient(120deg, #0f766e, #0284c7);
    padding: 30px;
    border-radius: 24px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 10px 30px rgba(2,132,199,0.25);
}
.hero h1 {
    margin: 0;
    font-size: 42px;
    font-weight: 900;
}
.hero p {
    font-size: 17px;
}
.card {
    background: rgba(255,255,255,0.96);
    padding: 25px;
    border-radius: 22px;
    box-shadow: 0 8px 28px rgba(15, 23, 42, 0.08);
    border: 1px solid #e2e8f0;
    margin-bottom: 22px;
}
.section-title {
    font-size: 24px;
    font-weight: 900;
    color: #0f172a;
    margin-bottom: 15px;
}
.subtle {
    color: #64748b;
    font-size: 15px;
}
.risk-low {
    background: #dcfce7;
    color: #166534;
    padding: 18px;
    border-radius: 16px;
    font-size: 20px;
    font-weight: 800;
}
.risk-moderate {
    background: #fef3c7;
    color: #92400e;
    padding: 18px;
    border-radius: 16px;
    font-size: 20px;
    font-weight: 800;
}
.risk-high {
    background: #fee2e2;
    color: #991b1b;
    padding: 18px;
    border-radius: 16px;
    font-size: 20px;
    font-weight: 800;
}
.stButton button {
    background: linear-gradient(90deg, #0891b2, #0f766e);
    color: white;
    border-radius: 14px;
    padding: 0.75rem 1.4rem;
    font-weight: 800;
    border: none;
}
.stDownloadButton button {
    background: #0f766e;
    color: white;
    border-radius: 14px;
    font-weight: 800;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #0f172a);
}
[data-testid="stSidebar"] * {
    color: white;
}
[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border-radius: 18px;
    box-shadow: 0 6px 20px rgba(15, 23, 42, 0.08);
}
</style>
""", unsafe_allow_html=True)

# ---------------- FOLDERS ----------------
os.makedirs("uploaded_reports", exist_ok=True)
os.makedirs("generated_reports", exist_ok=True)

# ---------------- LOAD MODEL ----------------
try:
    model = joblib.load("models/model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    imputer = joblib.load("models/imputer.pkl")
except Exception:
    st.error("Model files missing. Keep model.pkl, scaler.pkl, imputer.pkl inside models folder.")
    st.stop()

# ---------------- DATABASE ----------------
conn = sqlite3.connect("patients.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    patient_id TEXT,
    patient_name TEXT,
    phone TEXT,
    gender TEXT,
    doctor_name TEXT,
    glucose REAL,
    bp REAL,
    skin REAL,
    insulin REAL,
    bmi REAL,
    dpf REAL,
    age REAL,
    risk REAL,
    risk_level TEXT,
    uploaded_report TEXT
)
""")
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT UNIQUE,
    password TEXT,
    role TEXT
)
""")

conn.commit()

# ---------------- FUNCTIONS ----------------
def predict_risk(values):
    arr = np.array([values])
    arr = imputer.transform(arr)
    arr = scaler.transform(arr)
    prob = model.predict_proba(arr)[0][1]
    risk = float(prob * 100)

    if risk > 70:
        level = "HIGH"
    elif risk > 30:
        level = "MODERATE"
    else:
        level = "LOW"

    return risk, level


def save_patient(record):
    cursor.execute("""
    INSERT INTO patients (
        timestamp, patient_id, patient_name, phone, gender, doctor_name,
        glucose, bp, skin, insulin, bmi, dpf, age,
        risk, risk_level, uploaded_report
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, record)
    conn.commit()


def load_patients():
    return pd.read_sql_query("SELECT * FROM patients ORDER BY id DESC", conn)


def generate_pdf(patient_id, patient_name, phone, gender, doctor_name, age, glucose, bp, bmi, risk, level):
    safe_name = patient_name.replace(" ", "_")
    filename = f"generated_reports/{safe_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    pdf = FPDF()
    pdf.add_page()

    pdf.set_fill_color(15, 118, 110)
    pdf.rect(0, 0, 210, 32, "F")

    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 16, "MediPredict AI - Clinical Diabetes Report", ln=True, align="C")

    pdf.ln(18)
    pdf.set_text_color(0, 0, 0)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Patient Information", ln=True)

    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, f"Patient ID: {patient_id}", ln=True)
    pdf.cell(0, 8, f"Patient Name: {patient_name}", ln=True)
    pdf.cell(0, 8, f"Phone: {phone}", ln=True)
    pdf.cell(0, 8, f"Gender: {gender}", ln=True)
    pdf.cell(0, 8, f"Age: {age}", ln=True)
    pdf.cell(0, 8, f"Doctor: {doctor_name}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Clinical Parameters", ln=True)

    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, f"Glucose: {glucose}", ln=True)
    pdf.cell(0, 8, f"Blood Pressure: {bp}", ln=True)
    pdf.cell(0, 8, f"BMI: {bmi}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "AI Risk Assessment", ln=True)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, f"Risk Score: {risk:.2f}%", ln=True)
    pdf.cell(0, 8, f"Risk Level: {level}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "", 11)

    if level == "HIGH":
        recommendation = "Immediate consultation, HbA1c testing, fasting glucose test, and lifestyle intervention are recommended."
    elif level == "MODERATE":
        recommendation = "Further testing, glucose monitoring, diet control, and follow-up consultation are advised."
    else:
        recommendation = "Routine monitoring, healthy diet, and regular exercise are recommended."

    pdf.multi_cell(0, 8, f"Clinical Recommendation: {recommendation}")

    pdf.ln(5)
    pdf.set_font("Arial", "I", 10)
    pdf.multi_cell(0, 8, "Disclaimer: This is an AI-assisted screening tool and not a final medical diagnosis.")

    pdf.output(filename)
    return filename


def clinical_bot(user_msg):
    if chat_model is None:
        return "Gemini API key is not configured. Set GEMINI_API_KEY in environment variables."

    prompt = f"""
    You are an AI medical assistant inside a diabetes clinical decision support system.
    Explain medical information in very simple language for patients.
    Be professional, safe, and helpful.
    Do not give final diagnosis. Always advise doctor consultation for serious concerns.

    User question:
    {user_msg}
    """

    try:
        response = chat_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Error: {str(e)}"

# ---------------- SYMPTOM CHECKER ----------------
def symptom_checker(symptoms, age, gender):

    text = " ".join(symptoms).lower()

    emergency_keywords = [
        "chest pain",
        "shortness of breath",
        "left arm pain",
        "jaw pain",
        "cold sweat",
        "fainting",
        "severe headache",
        "speech problem",
        "face drooping",
        "sudden weakness"
    ]

    diabetes_keywords = [
        "frequent urination",
        "excessive thirst",
        "weight loss",
        "blurred vision",
        "fatigue",
        "slow healing wounds"
    ]

    heart_keywords = [
        "chest pain",
        "shortness of breath",
        "palpitations",
        "left arm pain",
        "jaw pain",
        "swelling legs"
    ]

    bp_keywords = [
        "headache",
        "dizziness",
        "high bp",
        "blurred vision"
    ]

    emergency = any(word in text for word in emergency_keywords)

    diabetes_score = sum(word in text for word in diabetes_keywords)
    heart_score = sum(word in text for word in heart_keywords)
    bp_score = sum(word in text for word in bp_keywords)

    if emergency:
        risk = "EMERGENCY"
        advice = "Emergency warning signs detected. Seek urgent medical help immediately."

    elif heart_score >= 2:
        risk = "HIGH HEART RISK"
        advice = "Heart-related symptoms detected. Consult cardiologist."

    elif diabetes_score >= 2:
        risk = "POSSIBLE DIABETES RISK"
        advice = "Symptoms may suggest diabetes risk. Take glucose and HbA1c tests."

    elif bp_score >= 2:
        risk = "POSSIBLE BP RISK"
        advice = "Symptoms may suggest blood pressure related issue."

    else:
        risk = "LOW / GENERAL"
        advice = "No major emergency pattern detected."

    return risk, advice

# ---------------- LOGIN / SIGNUP FUNCTIONS ----------------
def signup_user(name, email, password, role):
    try:
        cursor.execute("""
        INSERT INTO users (name, email, password, role)
        VALUES (?, ?, ?, ?)
        """, (name, email, password, role))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def login_user(email, password):
    cursor.execute("""
    SELECT name, email, role FROM users
    WHERE email = ? AND password = ?
    """, (email, password))
    return cursor.fetchone()


def login_page():
    st.markdown("""
    <div class="hero">
        <h1>🏥 MediPredict AI</h1>
        <p>Secure AI Clinical Decision Support Platform</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.3, 1])

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])

        with tab1:
            st.subheader("Login to Dashboard")

            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")

            if st.button("Login", use_container_width=True):
                user = login_user(email, password)

                if user:
                    st.session_state.logged_in = True
                    st.session_state.user_name = user[0]
                    st.session_state.user_email = user[1]
                    st.session_state.user_role = user[2]
                    st.success("Login successful")
                    st.rerun()
                else:
                    st.error("Invalid email or password")

        with tab2:
            st.subheader("Create New Account")

            name = st.text_input("Full Name", key="signup_name")
            signup_email = st.text_input("Email", key="signup_email")
            signup_password = st.text_input("Password", type="password", key="signup_password")
            role = st.selectbox("Role", ["Doctor", "Admin", "Receptionist"])

            if st.button("Create Account", use_container_width=True):
                if name.strip() == "" or signup_email.strip() == "" or signup_password.strip() == "":
                    st.warning("Please fill all fields.")
                else:
                    created = signup_user(name, signup_email, signup_password, role)

                    if created:
                        st.success("Account created successfully. Now login.")
                    else:
                        st.error("Email already exists.")

        st.markdown('</div>', unsafe_allow_html=True)


# ---------------- LOGIN CHECK ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login_page()
    st.stop()

# ---------------- ROLE BASED SIDEBAR ----------------
st.sidebar.markdown("## 🏥 MediPredict AI")
st.sidebar.markdown("Clinical Decision Support")

st.sidebar.markdown("---")
st.sidebar.write(f"👤 {st.session_state.user_name}")
st.sidebar.write(f"Role: {st.session_state.user_role}")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

st.sidebar.markdown("---")

role = st.session_state.user_role

if role == "Admin":
    allowed_pages = [
        "Doctor Dashboard",
        "Patient Records",
        "AI Health Assistant",
        "AI Chatbot",
        "Manage Users"
    ]

elif role == "Doctor":
    allowed_pages = [
        "Patient Screening",
        "Doctor Dashboard",
        "Patient Records",
        "AI Health Assistant",
        "AI Chatbot"
    ]

else:  # Receptionist
    allowed_pages = [
        "Patient Screening",
        "Patient Records"
    ]

page = st.sidebar.radio("Navigation", allowed_pages)

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>🏥 MediPredict AI</h1>
    <p>AI-powered Diabetes Clinical Decision Support System with patient-friendly disease explanation.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- PATIENT SCREENING ----------------
if page == "Patient Screening":

    left, right = st.columns([1.3, 0.7])

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">👤 Patient Registration</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)

        with c1:
            patient_id = st.text_input("Patient ID", value=f"PID-{datetime.datetime.now().strftime('%H%M%S')}")
            patient_name = st.text_input("Full Name")

        with c2:
            phone = st.text_input("Phone Number")
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])

        with c3:
            doctor_name = st.text_input("Doctor Name", value="Dr. ")
            age = st.number_input("Age", min_value=0.0, max_value=120.0)

        uploaded_file = st.file_uploader("Upload Lab Report / Medical Report", type=["pdf", "png", "jpg", "jpeg"])

        uploaded_path = ""
        if uploaded_file is not None:
            uploaded_path = os.path.join("uploaded_reports", uploaded_file.name)
            with open(uploaded_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success("Report uploaded successfully.")

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🧪 Clinical Parameters</div>', unsafe_allow_html=True)

        a, b, c = st.columns(3)

        with a:
            glucose = st.number_input("Glucose", min_value=0.0)
            bp = st.number_input("Blood Pressure", min_value=0.0)

        with b:
            skin = st.number_input("Skin Thickness", min_value=0.0)
            insulin = st.number_input("Insulin", min_value=0.0)

        with c:
            bmi = st.number_input("BMI", min_value=0.0)
            dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)

        analyze = st.button("🔍 Analyze Patient Risk")

        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📋 Patient-Friendly Result</div>', unsafe_allow_html=True)
        st.write("This app explains diabetes risk using simple charts:")
        st.write("• Risk meter")
        st.write("• Glucose comparison")
        st.write("• BMI category")
        st.write("• Risk factor chart")
        st.write("• Simple health explanation")
        st.markdown('</div>', unsafe_allow_html=True)

    if analyze:
        if patient_name.strip() == "":
            st.error("Please enter patient name.")
        else:
            values = [glucose, bp, skin, insulin, bmi, dpf, age]
            risk, level = predict_risk(values)

            save_patient((
                str(datetime.datetime.now()),
                patient_id,
                patient_name,
                phone,
                gender,
                doctor_name,
                glucose,
                bp,
                skin,
                insulin,
                bmi,
                dpf,
                age,
                risk,
                level,
                uploaded_path
            ))

            st.markdown("## 🧠 AI Risk Result")

            r1, r2, r3 = st.columns(3)
            r1.metric("Risk Score", f"{risk:.2f}%")
            r2.metric("Risk Level", level)
            r3.metric("Patient", patient_name)

            if level == "HIGH":
                st.markdown('<div class="risk-high">HIGH RISK — Immediate medical attention recommended</div>', unsafe_allow_html=True)
            elif level == "MODERATE":
                st.markdown('<div class="risk-moderate">MODERATE RISK — Further testing and follow-up advised</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="risk-low">LOW RISK — Routine monitoring recommended</div>', unsafe_allow_html=True)

            st.markdown("## 📊 Easy Disease Understanding")

            g1, g2 = st.columns(2)

            with g1:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk,
                    title={"text": "Diabetes Risk Meter"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#0284c7"},
                        "steps": [
                            {"range": [0, 30], "color": "#dcfce7"},
                            {"range": [30, 70], "color": "#fef3c7"},
                            {"range": [70, 100], "color": "#fee2e2"}
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": risk
                        }
                    }
                ))
                fig_gauge.update_layout(height=330)
                st.plotly_chart(fig_gauge, use_container_width=True)

            with g2:
                glucose_status = "Normal" if glucose < 100 else "Borderline" if glucose < 126 else "High"
                bmi_status = "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### 🧾 Simple Health Summary")
                st.write(f"**Glucose Status:** {glucose_status}")
                st.write(f"**BMI Status:** {bmi_status}")
                st.write(f"**Diabetes Risk Level:** {level}")

                if level == "HIGH":
                    st.error("Your risk is high. Please consult a doctor and take confirmatory tests.")
                elif level == "MODERATE":
                    st.warning("Your risk is moderate. Lifestyle improvement and monitoring are advised.")
                else:
                    st.success("Your risk is low. Continue healthy habits and routine checkups.")

                st.markdown('</div>', unsafe_allow_html=True)

            risk_factors = {
                "Glucose": min(glucose / 200 * 100, 100),
                "BMI": min(bmi / 40 * 100, 100),
                "Age": min(age / 80 * 100, 100),
                "Blood Pressure": min(bp / 160 * 100, 100)
            }

            factor_df = pd.DataFrame({
                "Factor": list(risk_factors.keys()),
                "Impact": list(risk_factors.values())
            })

            fig_bar = px.bar(
                factor_df,
                x="Factor",
                y="Impact",
                text="Impact",
                title="Main Factors Affecting Your Diabetes Risk"
            )
            fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_bar.update_layout(yaxis_range=[0, 110], height=400)
            st.plotly_chart(fig_bar, use_container_width=True)

            range_df = pd.DataFrame({
                "Parameter": ["Glucose", "BMI", "Blood Pressure"],
                "Your Value": [glucose, bmi, bp],
                "Healthy Reference": [100, 25, 120]
            })

            fig_compare = px.bar(
                range_df,
                x="Parameter",
                y=["Your Value", "Healthy Reference"],
                barmode="group",
                title="Your Values Compared With Healthy Reference"
            )
            st.plotly_chart(fig_compare, use_container_width=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 🧠 What This Means in Simple Words")

            if glucose >= 126:
                st.write("• Your glucose is above the usual healthy range, which can increase diabetes risk.")
            elif glucose >= 100:
                st.write("• Your glucose is slightly above normal and should be monitored.")
            else:
                st.write("• Your glucose is within a healthy range.")

            if bmi >= 30:
                st.write("• Your BMI is in the obese range, which can increase insulin resistance.")
            elif bmi >= 25:
                st.write("• Your BMI is in the overweight range, so weight control may help.")
            else:
                st.write("• Your BMI is within a healthier range.")

            if bp >= 130:
                st.write("• Your blood pressure is high, which may increase overall metabolic risk.")
            else:
                st.write("• Your blood pressure is within a safer range.")

            st.write("• This is only a screening result. A doctor should confirm it with tests like HbA1c.")
            st.markdown('</div>', unsafe_allow_html=True)

            pdf_path = generate_pdf(
                patient_id, patient_name, phone, gender,
                doctor_name, age, glucose, bp, bmi, risk, level
            )

            with open(pdf_path, "rb") as file:
                st.download_button(
                    "📄 Download Clinical PDF Report",
                    data=file,
                    file_name=os.path.basename(pdf_path),
                    mime="application/pdf"
                )

# ---------------- DOCTOR DASHBOARD ----------------
elif page == "Doctor Dashboard":

    df = load_patients()

    if df.empty:
        st.info("No patient records yet. Add patients from Patient Screening page.")
    else:
        total = len(df)
        high = len(df[df["risk_level"] == "HIGH"])
        moderate = len(df[df["risk_level"] == "MODERATE"])
        low = len(df[df["risk_level"] == "LOW"])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Patients", total)
        m2.metric("High Risk", high)
        m3.metric("Moderate Risk", moderate)
        m4.metric("Low Risk", low)

        st.markdown("---")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Risk Distribution")
            fig = px.pie(df, names="risk_level", hole=0.45)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Risk Score Trend")
            fig = px.line(df.sort_values("id"), x="timestamp", y="risk", markers=True)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Glucose vs Risk Analysis")
        fig = px.scatter(
            df,
            x="glucose",
            y="risk",
            color="risk_level",
            size="bmi",
            hover_data=["patient_name", "age", "bp"]
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Recent Patients")
        st.dataframe(df.head(10), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PATIENT RECORDS ----------------
elif page == "Patient Records":

    df = load_patients()

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📁 Patient Records</div>', unsafe_allow_html=True)

    if df.empty:
        st.info("No records found.")
    else:
        search = st.text_input("Search by patient name, ID, or phone")

        if search:
            df = df[
                df["patient_name"].str.contains(search, case=False, na=False) |
                df["patient_id"].str.contains(search, case=False, na=False) |
                df["phone"].str.contains(search, case=False, na=False)
            ]

        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Patient Records CSV",
            data=csv,
            file_name="patient_records.csv",
            mime="text/csv"
        )

    st.markdown('</div>', unsafe_allow_html=True)
# ---------------- AI HEALTH ASSISTANT ----------------
elif page == "AI Health Assistant":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🩺 AI Symptom-Based Health Assistant</div>', unsafe_allow_html=True)

    st.write("Select symptoms and AI will guide patient to next step.")

    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        patient_age = st.number_input(
            "Patient Age",
            min_value=0,
            max_value=120
        )

        patient_gender = st.selectbox(
            "Gender",
            ["Male", "Female", "Other"]
        )

    with col2:

        symptoms = st.multiselect(
            "Select Symptoms",
            [
                "Chest pain",
                "Shortness of breath",
                "Left arm pain",
                "Jaw pain",
                "Cold sweat",
                "Palpitations",
                "Frequent urination",
                "Excessive thirst",
                "Blurred vision",
                "Fatigue",
                "Weight loss",
                "Slow healing wounds",
                "Headache",
                "Dizziness",
                "High BP",
                "Swelling legs",
                "Severe headache",
                "Speech problem",
                "Face drooping",
                "Sudden weakness"
            ]
        )

    if st.button("Analyze Symptoms"):

        if not symptoms:
            st.warning("Please select symptoms.")

        else:

            risk, advice = symptom_checker(
                symptoms,
                patient_age,
                patient_gender
            )

            st.markdown("## 🧠 AI Symptom Analysis")

            if risk == "EMERGENCY":
                st.error(risk)

            elif "HIGH" in risk:
                st.warning(risk)

            else:
                st.info(risk)

            st.write(advice)

            # -------- AI EXPLANATION --------
            prompt = f"""
            You are an AI healthcare assistant.

            Explain the following symptom analysis
            in simple patient-friendly language.

            Age: {patient_age}
            Gender: {patient_gender}
            Symptoms: {symptoms}
            Risk: {risk}
            Advice: {advice}

            Explain:
            1. What these symptoms may indicate
            2. What patient should do next
            3. Mention emergency warning signs
            4. Mention this is not final diagnosis
            """

            with st.spinner("AI is analyzing symptoms..."):

                response = chat_model.generate_content(prompt)

                st.markdown("## 🤖 AI Explanation")

                st.info(response.text)

# ---------------- MANAGE USERS ----------------
elif page == "Manage Users":

    if st.session_state.user_role != "Admin":
        st.error("Access denied. Admin only.")
    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">👥 Manage Users</div>', unsafe_allow_html=True)

        users_df = pd.read_sql_query(
            "SELECT id, name, email, role FROM users ORDER BY id DESC",
            conn
        )

        st.dataframe(users_df, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

# ---------------- AI CHATBOT ----------------
elif page == "AI Chatbot":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🤖 MediPredict AI Clinical Assistant</div>', unsafe_allow_html=True)
    st.markdown('<p class="subtle">Ask about glucose, BMI, diabetes risk level, reports, or recommendations.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.write(chat["message"])

    user_input = st.chat_input("Ask MediPredict AI...")

    if user_input:
        st.session_state.chat_history.append({
            "role": "user",
            "message": user_input
        })

        response = clinical_bot(user_input)

        st.session_state.chat_history.append({
            "role": "assistant",
            "message": response
        })

        st.rerun()


st.markdown("""
<hr>
<p style='text-align:center; color:gray;'>
© 2026 MediPredict AI | Developed by Isha Dave. All Rights Reserved.
</p>
""", unsafe_allow_html=True)
