import streamlit as st
import pickle
import pandas as pd
import os

# ==========================================
# 1. Page Configuration & Navigation State
# ==========================================
st.set_page_config(
    page_title="AI Medical Diagnosis",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="collapsed" # Hides sidebar completely
)

# Initialize session state for navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Home'

def navigate(page_name):
    st.session_state.current_page = page_name

# ==========================================
# 2. Modern UI/UX CSS Injection
# ==========================================
css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    /* Global Font & Background */
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif !important;
    }
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* Force Dark Text for Readability */
    .stApp, .stApp p, .stApp span, .stApp label, .stApp div,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: #1E293B;
    }

    /* Hide Sidebar & Default Streamlit UI */
    [data-testid="collapsedControl"] { display: none !important; }
    [data-testid="stSidebar"] { display: none !important; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Hide Form Borders to maintain the clean design */
    [data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
        background-color: transparent !important;
    }

    /* Modern Disease Cards (Home Page) */
    .disease-card {
        background-color: #FFFFFF;
        border-radius: 20px;
        padding: 30px 20px;
        text-align: center;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 10px;
        height: 100%;
    }
    .disease-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.1);
        border-color: #BFDBFE;
    }
    .card-icon {
        font-size: 3.5rem;
        margin-bottom: 15px;
    }
    .card-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #0F172A !important;
        margin-bottom: 10px;
    }
    .card-desc {
        font-size: 0.9rem;
        color: #64748B !important;
        margin-bottom: 20px;
        line-height: 1.5;
    }

    /* Input Field Styling */
    div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
        border-radius: 12px !important;
        border: 1px solid #CBD5E1 !important;
        background-color: #FFFFFF !important;
    }

    /* FIX: Force input and select text to be dark so it is visible on white background */
    input, textarea, div[data-baseweb="select"] * {
        color: #1E293B !important;
        -webkit-text-fill-color: #1E293B !important;
    }

    /* Primary Button (Blue Gradient for Predict/Start) */
    button[kind="primary"], button[kind="primaryFormSubmit"] {
        background: linear-gradient(135deg, #2563EB, #1D4ED8) !important;
        color: #FFFFFF !important;
        border-radius: 12px !important;
        padding: 0.6rem 2rem !important;
        border: none !important;
        font-weight: 500 !important;
        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.2) !important;
        transition: all 0.3s ease !important;
    }
    button[kind="primary"] *, button[kind="primaryFormSubmit"] * {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }
    button[kind="primary"]:hover, button[kind="primaryFormSubmit"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 15px rgba(37, 99, 235, 0.3) !important;
    }

    /* Secondary Button (Back Button) */
    button[kind="secondary"] {
        background-color: #FFFFFF !important;
        color: #334155 !important;
        border-radius: 12px !important;
        border: 1px solid #CBD5E1 !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }
    button[kind="secondary"] * {
        color: #334155 !important;
        -webkit-text-fill-color: #334155 !important;
    }
    button[kind="secondary"]:hover {
        background-color: #F1F5F9 !important;
        border-color: #94A3B8 !important;
    }

    /* Result Cards */
    .result-success {
        background-color: #ECFDF5 !important;
        border-left: 6px solid #10B981 !important;
        padding: 24px !important;
        border-radius: 12px !important;
        margin-top: 25px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02) !important;
    }
    .result-success, .result-success * {
        color: #065F46 !important;
        -webkit-text-fill-color: #065F46 !important;
    }
    
    .result-danger {
        background-color: #FEF2F2 !important;
        border-left: 6px solid #EF4444 !important;
        padding: 24px !important;
        border-radius: 12px !important;
        margin-top: 25px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02) !important;
    }
    .result-danger, .result-danger * {
        color: #991B1B !important;
        -webkit-text-fill-color: #991B1B !important;
    }
    
    .page-title-box {
        margin-bottom: 2rem;
    }
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# ==========================================
# 3. Model Loading
# ==========================================
@st.cache_resource
def load_all_models():
    base_path = "./Models"

    return {
        "diabetes": pickle.load(open(os.path.join(base_path, "diabetes_model.sav"), "rb")),
        "heart_disease": pickle.load(open(os.path.join(base_path, "heart_disease_model.sav"), "rb")),
        "heart_scaler": pickle.load(open(os.path.join(base_path, "scaler.sav"), "rb")),
        "parkinsons": pickle.load(open(os.path.join(base_path, "parkinsons_model.sav"), "rb")),
        "lung_cancer": pickle.load(open(os.path.join(base_path, "lungs_disease_model.sav"), "rb")),
        "thyroid": pickle.load(open(os.path.join(base_path, "thyroid_model.sav"), "rb")),
    }

models = load_all_models()

# Helper function to display results beautifully
def display_result(is_disease, positive_text, negative_text):
    if is_disease:
        st.markdown(f"""
            <div class="result-danger">
                <div style="font-size: 1.25rem; font-weight: 600; margin-bottom: 8px;">
                    ⚠️ Higher-Risk Pattern Flagged
                </div>
                <div style="font-size: 1rem;">{positive_text}</div>
                <hr style="border-color: rgba(239,68,68,0.2); margin: 15px 0;">
                <div style="font-size: 0.9rem;">
                    <strong>Recommendation:</strong>
                    Consult a qualified healthcare professional for further evaluation.
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="result-success">
                <div style="font-size: 1.25rem; font-weight: 600; margin-bottom: 8px;">
                    ✅ No Concerning Pattern Flagged
                </div>
                <div style="font-size: 1rem;">{negative_text}</div>
                <hr style="border-color: rgba(16,185,129,0.2); margin: 15px 0;">
                <div style="font-size: 0.9rem;">
                    <strong>Recommendation:</strong>
                    Continue routine healthcare check-ups as appropriate.
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.info(
        "This tool provides preliminary model-based screening support only. "
        "It is not a clinical diagnosis and should not replace professional medical advice."
    )


# ==========================================
# 4. Routing Logic
# ==========================================

# ------------------------------------------
# HOME PAGE
# ------------------------------------------
if st.session_state.current_page == 'Home':
    st.markdown("""
        <div class="page-title-box" style="text-align: center;">
            <h1 style="font-weight: 700;">🩺 AI Medical Diagnosis Assistant</h1>
            <p style="font-size: 1.1rem; color: #64748B;">Select a disease to begin prediction.</p>
        </div>
    """, unsafe_allow_html=True)

    # First Row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="disease-card">
                <div class="card-icon">🩸</div>
                <div class="card-title">Diabetes</div>
                <div class="card-desc">Predict diabetes risk based on metabolic factors, glucose levels, and BMI.</div>
            </div>
        """, unsafe_allow_html=True)
        st.button("Start Prediction", key="btn_diab", type="primary", use_container_width=True, on_click=navigate, args=('Diabetes',))

    with col2:
        st.markdown("""
            <div class="disease-card">
                <div class="card-icon">❤️</div>
                <div class="card-title">Heart Disease</div>
                <div class="card-desc">Evaluate cardiovascular metrics to assess the likelihood of heart conditions.</div>
            </div>
        """, unsafe_allow_html=True)
        st.button("Start Prediction", key="btn_heart", type="primary", use_container_width=True, on_click=navigate, args=('Heart Disease',))

    with col3:
        st.markdown("""
            <div class="disease-card">
                <div class="card-icon">🧠</div>
                <div class="card-title">Parkinson's Disease</div>
                <div class="card-desc">Detect Parkinson's disease using advanced acoustic and vocal biomarkers.</div>
            </div>
        """, unsafe_allow_html=True)
        st.button("Start Prediction", key="btn_park", type="primary", use_container_width=True, on_click=navigate, args=("Parkinson's",))

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Second Row
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.markdown("""
            <div class="disease-card">
                <div class="card-icon">🫁</div>
                <div class="card-title">Lung Cancer</div>
                <div class="card-desc">Analyze behavioral and physiological symptoms for early lung cancer detection.</div>
            </div>
        """, unsafe_allow_html=True)
        st.button("Start Prediction", key="btn_lung", type="primary", use_container_width=True, on_click=navigate, args=('Lung Cancer',))

    with col5:
        st.markdown("""
            <div class="disease-card">
                <div class="card-icon">🦋</div>
                <div class="card-title">Thyroid Disease</div>
                <div class="card-desc">Assess hormonal levels and clinical features for thyroid abnormalities.</div>
            </div>
        """, unsafe_allow_html=True)
        st.button("Start Prediction", key="btn_thyroid", type="primary", use_container_width=True, on_click=navigate, args=('Thyroid',))
        
    with col6:
        pass

# ------------------------------------------
# DIABETES PAGE
# ------------------------------------------
elif st.session_state.current_page == 'Diabetes':
    st.button("← Back to Home", on_click=navigate, args=('Home',))
    
    st.markdown("""
        <div class="page-title-box">
            <h2>🩸 Diabetes Prediction</h2>
            <p>Enter the patient's metabolic details below to predict diabetes risk.</p>
        </div>
    """, unsafe_allow_html=True)

    with st.form("diabetes_form"):
        col1, col2 = st.columns(2)
        with col1:
            Pregnancies = st.number_input('Number of Pregnancies', value=None, step=1, key="d_preg")
            BloodPressure = st.number_input('Blood Pressure (mmHg)', value=None, step=1, key="d_bp")
            Insulin = st.number_input('Insulin Level (IU/mL)', value=None, step=1, key="d_ins")
            DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', value=None, format="%f", key="d_dpf")
        with col2:
            Glucose = st.number_input('Glucose Level (mg/dL)', value=None, step=1, key="d_gluc")
            SkinThickness = st.number_input('Skin Thickness (mm)', value=None, step=1, key="d_skin")
            BMI = st.number_input('BMI Value', value=None, format="%f", key="d_bmi")
            Age = st.number_input('Age', value=None, step=1, key="d_age")
        
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button('Predict Diabetes Risk', type="primary", use_container_width=True)
        
    if submitted:
        diabetes_values = [
            Pregnancies, Glucose, BloodPressure, SkinThickness,
            Insulin, BMI, DiabetesPedigreeFunction, Age
        ]

        if any(value is None for value in diabetes_values):
            st.warning("Please complete all fields before making a prediction.")
        else:
            input_data = pd.DataFrame(
                [diabetes_values],
                columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
            )
            prediction = models['diabetes'].predict(input_data)
            display_result(
                prediction[0] == 1,
                "The model indicates a higher-risk pattern for diabetes.",
                "The model did not identify a concerning pattern for diabetes."
            )

# ------------------------------------------
# HEART DISEASE PAGE
# ------------------------------------------
elif st.session_state.current_page == 'Heart Disease':
    st.button("← Back to Home", on_click=navigate, args=('Home',))
    
    st.markdown("""
        <div class="page-title-box">
            <h2>❤️ Heart Disease Prediction</h2>
            <p>Provide cardiovascular metrics to evaluate risk.</p>
        </div>
    """, unsafe_allow_html=True)

    with st.form("heart_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input('Age', value=None, step=1, key="h_age")
            cp = st.selectbox('Chest Pain Type (0-3)', options=[0, 1, 2, 3], index=None, key="h_cp")
            chol = st.number_input('Serum Cholesterol (mg/dl)', value=None, step=1, key="h_chol")
            restecg = st.selectbox('Resting ECG Results (0-2)', options=[0, 1, 2], index=None, key="h_restecg")
            exang = st.selectbox('Exercise Induced Angina', options=[1, 0], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)", index=None, key="h_exang")
            slope = st.selectbox('Slope of Peak Exercise ST Segment', options=[0, 1, 2], index=None, key="h_slope")
            thal = st.selectbox('Thal', options=[0, 1, 2, 3], index=None, key="h_thal")
        with col2:
            sex = st.selectbox('Sex', options=[1, 0], format_func=lambda x: "Male (1)" if x == 1 else "Female (0)", index=None, key="h_sex")
            trestbps = st.number_input('Resting Blood Pressure (mmHg)', value=None, step=1, key="h_trestbps")
            fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[1, 0], format_func=lambda x: "True (1)" if x == 1 else "False (0)", index=None, key="h_fbs")
            thalach = st.number_input('Maximum Heart Rate Achieved', value=None, step=1, key="h_thalach")
            oldpeak = st.number_input('ST Depression Induced by Exercise', value=None, format="%f", key="h_oldpeak")
            ca = st.selectbox('Major Vessels Colored by Fluoroscopy (0-3)', options=[0, 1, 2, 3], index=None, key="h_ca")

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button('Predict Heart Disease Risk', type="primary", use_container_width=True)
        
    if submitted:
        heart_values = [
            age, sex, cp, trestbps, chol, fbs, restecg,
            thalach, exang, oldpeak, slope, ca, thal
        ]

        if any(value is None for value in heart_values):
            st.warning("Please complete all fields before making a prediction.")
        else:
            input_data = pd.DataFrame(
                [heart_values],
                columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            )
            scaled_input = models['heart_scaler'].transform(input_data)
            prediction = models['heart_disease'].predict(scaled_input)
            display_result(
                prediction[0] == 1,
                "The model indicates a higher-risk pattern for heart disease.",
                "The model did not identify a concerning pattern for heart disease."
            )

# ------------------------------------------
# PARKINSON'S PAGE
# ------------------------------------------
elif st.session_state.current_page == "Parkinson's":
    st.button("← Back to Home", on_click=navigate, args=('Home',))
    
    st.markdown("""
        <div class="page-title-box">
            <h2>🧠 Parkinson's Disease Prediction</h2>
            <p>Analyze vocal biomarkers to detect Parkinson's disease.</p>
        </div>
    """, unsafe_allow_html=True)

    with st.form("parkinsons_form"):
        col1, col2 = st.columns(2)
        with col1:
            fo = st.number_input('MDVP:Fo(Hz)', value=119.99, format="%f", key="p_fo")
            flo = st.number_input('MDVP:Flo(Hz)', value=74.99, format="%f", key="p_flo")
            Jitter_Abs = st.number_input('MDVP:Jitter(Abs)', value=0.00007, format="%f", key="p_jit_abs")
            PPQ = st.number_input('MDVP:PPQ', value=0.00554, format="%f", key="p_ppq")
            Shimmer = st.number_input('MDVP:Shimmer', value=0.04374, format="%f", key="p_shimmer")
            APQ3 = st.number_input('Shimmer:APQ3', value=0.02182, format="%f", key="p_apq3")
            APQ = st.number_input('MDVP:APQ', value=0.02971, format="%f", key="p_apq")
            NHR = st.number_input('NHR', value=0.02211, format="%f", key="p_nhr")
            RPDE = st.number_input('RPDE', value=0.41478, format="%f", key="p_rpde")
            spread1 = st.number_input('Spread1', value=-4.81303, format="%f", key="p_spread1")
            D2 = st.number_input('D2', value=2.30144, format="%f", key="p_d2")
        with col2:
            fhi = st.number_input('MDVP:Fhi(Hz)', value=157.30, format="%f", key="p_fhi")
            Jitter_percent = st.number_input('MDVP:Jitter(%)', value=0.00784, format="%f", key="p_jit_per")
            RAP = st.number_input('MDVP:RAP', value=0.0037, format="%f", key="p_rap")
            DDP = st.number_input('Jitter:DDP', value=0.01109, format="%f", key="p_ddp")
            Shimmer_dB = st.number_input('MDVP:Shimmer(dB)', value=0.426, format="%f", key="p_shimdb")
            APQ5 = st.number_input('Shimmer:APQ5', value=0.03130, format="%f", key="p_apq5")
            DDA = st.number_input('Shimmer:DDA', value=0.06545, format="%f", key="p_dda")
            HNR = st.number_input('HNR', value=21.033, format="%f", key="p_hnr")
            DFA = st.number_input('DFA', value=0.81528, format="%f", key="p_dfa")
            spread2 = st.number_input('Spread2', value=0.26648, format="%f", key="p_spread2")
            PPE = st.number_input('PPE', value=0.28465, format="%f", key="p_ppe")

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Predict Parkinson's Risk", type="primary", use_container_width=True)
        
    if submitted:
        input_data = pd.DataFrame(
            [[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]],
            columns=['fo', 'fhi', 'flo', 'Jitter_percent', 'Jitter_Abs', 'RAP', 'PPQ', 'DDP', 'Shimmer', 'Shimmer_dB', 'APQ3', 'APQ5', 'APQ', 'DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']
        )
        prediction = models['parkinsons'].predict(input_data)
        display_result(prediction[0] == 1, "The model indicates the presence of Parkinson's disease.", "The model indicates no presence of Parkinson's disease.")

# ------------------------------------------
# LUNG CANCER PAGE
# ------------------------------------------
elif st.session_state.current_page == "Lung Cancer":

    st.button("← Back to Home", on_click=navigate, args=('Home',))

    st.markdown("""
        <div class="page-title-box">
            <h2>🫁 Lung Cancer Prediction</h2>
            <p>Analyze behavioral and physiological symptoms for early lung cancer detection.</p>
        </div>
    """, unsafe_allow_html=True)

    yn = lambda x: "Yes (1)" if x == 1 else "No (0)"

    with st.form("lung_form"):

        col1, col2 = st.columns(2)

        with col1:

            GENDER = st.selectbox("Gender", [1,0],
                                  format_func=lambda x: "Male" if x==1 else "Female")

            SMOKING = st.selectbox("Smoking",[1,0],format_func=yn)

            ANXIETY = st.selectbox("Anxiety",[1,0],format_func=yn)

            CHRONIC_DISEASE = st.selectbox("Chronic Disease",[1,0],format_func=yn)

            ALLERGY = st.selectbox("Allergy",[1,0],format_func=yn)

            ALCOHOL_CONSUMING = st.selectbox("Alcohol Consuming",[1,0],format_func=yn)

            SHORTNESS_OF_BREATH = st.selectbox("Shortness of Breath",[1,0],format_func=yn)

            CHEST_PAIN = st.selectbox("Chest Pain",[1,0],format_func=yn)

        with col2:

            AGE = st.number_input("Age",18,100,45)

            YELLOW_FINGERS = st.selectbox("Yellow Fingers",[1,0],format_func=yn)

            PEER_PRESSURE = st.selectbox("Peer Pressure",[1,0],format_func=yn)

            FATIGUE = st.selectbox("Fatigue",[1,0],format_func=yn)

            WHEEZING = st.selectbox("Wheezing",[1,0],format_func=yn)

            COUGHING = st.selectbox("Coughing",[1,0],format_func=yn)

            SWALLOWING_DIFFICULTY = st.selectbox("Swallowing Difficulty",[1,0],format_func=yn)

        submitted = st.form_submit_button(
            "Predict Lung Cancer Risk",
            type="primary",
            use_container_width=True
        )

    if submitted:

        # Input dictionary
        values = {
            "GENDER": GENDER,
            "AGE": AGE,
            "SMOKING": SMOKING,
            "YELLOW_FINGERS": YELLOW_FINGERS,
            "ANXIETY": ANXIETY,
            "PEER_PRESSURE": PEER_PRESSURE,
            "CHRONIC DISEASE": CHRONIC_DISEASE,
            "FATIGUE": FATIGUE,
            "ALLERGY": ALLERGY,
            "WHEEZING": WHEEZING,
            "ALCOHOL CONSUMING": ALCOHOL_CONSUMING,
            "COUGHING": COUGHING,
            "SHORTNESS OF BREATH": SHORTNESS_OF_BREATH,
            "SWALLOWING DIFFICULTY": SWALLOWING_DIFFICULTY,
            "CHEST PAIN": CHEST_PAIN,
        }

        model = models["lung_cancer"]

        # Model ke original feature names
        expected = list(model.feature_names_in_)

        # Agar model me trailing spaces hain to handle karega
        fixed = {}

        for col in expected:
            key = col.strip()
            for k in values:
                if k.strip() == key:
                    fixed[col] = values[k]
                    break

        input_data = pd.DataFrame([fixed])

        print("Model Features :", expected)
        print("Input Features :", input_data.columns.tolist())

        prediction = model.predict(input_data)

        if prediction[0] == 1:
            display_result(
                True,
                "The model indicates the presence of Lung Cancer.",
                ""
            )
        else:
            display_result(
                False,
                "",
                "The model indicates no presence of Lung Cancer."
            )
# ------------------------------------------
# THYROID PAGE
# ------------------------------------------
elif st.session_state.current_page == "Thyroid":
    st.button("← Back to Home", on_click=navigate, args=('Home',))
    
    st.markdown("""
        <div class="page-title-box">
            <h2>🦋 Thyroid Disease Prediction</h2>
            <p>Evaluate hormonal levels and clinical features for thyroid abnormalities.</p>
        </div>
    """, unsafe_allow_html=True)

    with st.form("thyroid_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input('Age', value=35, step=1, key="t_age")
            on_thyroxine = st.selectbox('On Thyroxine', options=[1, 0], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)", key="t_onthy")
            t3_measured = st.selectbox('T3 Measured', options=[1, 0], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)", key="t_t3m")
            tt4 = st.number_input('TT4 Level', value=100.0, format="%f", key="t_tt4")
        with col2:
            sex = st.selectbox('Sex', options=[1, 0], format_func=lambda x: "Male (1)" if x == 1 else "Female (0)", key="t_sex")
            tsh = st.number_input('TSH Level', value=1.5, format="%f", key="t_tsh")
            t3 = st.number_input('T3 Level', value=2.0, format="%f", key="t_t3")

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Predict Thyroid Risk", type="primary", use_container_width=True)
        
    if submitted:
        input_data = pd.DataFrame(
    [[age, sex, on_thyroxine, tsh, t3_measured, t3, tt4]],
    columns=[
        'age',
        'sex',
        'on thyroxine',
        'TSH',
        'T3 measured',
        'T3',
        'TT4'
    ]
)
        prediction = models['thyroid'].predict(input_data)
        display_result(prediction[0] == 1, "The model indicates the presence of Thyroid disease.", "The model indicates no presence of Thyroid disease.")