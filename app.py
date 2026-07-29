import pandas as pd
import streamlit as st
import joblib


st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)


st.markdown("""
<style>

body {
    background-color: #f5f7fb;
}

.main-header {
    background: linear-gradient(90deg,#ff4b4b,#ff758c);
    padding: 25px;
    border-radius: 15px;
    text-align:center;
    color:white;
    margin-bottom:25px;
}

.main-header h1 {
    font-size:40px;
    margin-bottom:5px;
}

.card {
    background:white;
    padding:25px;
    border-radius:15px;
    box-shadow:0px 5px 20px rgba(0,0,0,0.08);
    margin-bottom:20px;
}

.result-card {
    padding:25px;
    border-radius:15px;
    text-align:center;
    font-size:22px;
}

.stButton button {
    width:100%;
    height:50px;
    border-radius:12px;
    font-size:18px;
    background:#ff4b4b;
    color:white;
}

.stButton button:hover {
    background:#d93636;
}

</style>
""", unsafe_allow_html=True)


model = joblib.load("L.Regression_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

st.markdown("""
<div class="main-header">

<h1>❤️ Heart Disease Prediction System</h1>

<p>
AI-powered prediction system to estimate heart disease risk
</p>

</div>
""", unsafe_allow_html=True)





st.markdown(
"""
<div class="card">

<h2>👤 Patient Information</h2>

</div>
""",
unsafe_allow_html=True
)


col1,col2,col3 = st.columns(3)


with col1:

    age = st.slider(
        "Age",
        18,
        100,
        40
    )

    sex = st.selectbox(
        "Gender",
        ["M","F"]
    )

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ATA","NAP","TA","ASY"]
    )


with col2:

    resting_BP = st.number_input(
        "Resting Blood Pressure",
        80,
        200,
        120
    )


    cholesterol = st.number_input(
        "Cholesterol",
        100,
        600,
        200
    )


    fasting_bs = st.selectbox(
        "Fasting Blood Sugar",
        [0,1]
    )



with col3:

    resting_ECG = st.selectbox(
        "Resting ECG",
        ["Normal","ST","LVH"]
    )

    max_hr = st.slider(
        "Maximum Heart Rate",
        60,
        220,
        150
    )

    exercise_angina = st.selectbox(
        "Exercise Angina",
        ["Y","N"]
    )



col4,col5 = st.columns(2)


with col4:

    oldpeak = st.slider(
        "Oldpeak",
        0.0,
        6.0,
        1.0
    )


with col5:

    st_slope = st.selectbox(
        "ST Slope",
        ["UP","Flat","Down"]
    )



st.divider()




if st.button("🔍 Analyze Heart Risk"):


    raw_input = {

        "Age":age,
        "RestingBP":resting_BP,
        "Cholesterol":cholesterol,
        "FastingBS":fasting_bs,
        "MaxHR":max_hr,
        "Oldpeak":oldpeak,


        "Sex_"+sex:1,
        "ChestPainType_"+chest_pain:1,
        "RestingECG_"+resting_ECG:1,
        "ExerciseAngina_"+exercise_angina:1,
        "ST_Slope_"+st_slope:1

    }


    input_df = pd.DataFrame([raw_input])


    for col in expected_columns:

        if col not in input_df.columns:
            input_df[col]=0


    input_df = input_df[expected_columns]


    scaled_input = scaler.transform(input_df)


    prediction = model.predict(scaled_input)[0]


    probability=None

    if hasattr(model,"predict_proba"):

        probability = model.predict_proba(
            scaled_input
        )[0][1]



    

    st.subheader("📊 Prediction Result")


    if prediction == 1:


        st.markdown(
        """
        <div class="result-card"
        style="background:#ffe5e5;color:#b30000">

        ⚠️ High Risk of Heart Disease

        </div>
        """,
        unsafe_allow_html=True
        )


    else:


        st.markdown(
        """
        <div class="result-card"
        style="background:#e6ffe6;color:#006600">

        ✅ Low Risk of Heart Disease

        </div>
        """,
        unsafe_allow_html=True
        )



    if probability:


        st.progress(
            probability
        )

        st.write(
            f"### Risk Probability: {probability:.2%}"
        )





    with st.expander(
        "📋 View Patient Data"
    ):

        st.dataframe(
            input_df,
            use_container_width=True
        )



    st.info(
        """
        ⚕️ Disclaimer:
        This AI prediction is for educational purposes only.
        It is not a medical diagnosis. Always consult a healthcare professional.
        """
    )




st.divider()

st.caption(
    "Built with Python | Machine Learning | Streamlit"
)