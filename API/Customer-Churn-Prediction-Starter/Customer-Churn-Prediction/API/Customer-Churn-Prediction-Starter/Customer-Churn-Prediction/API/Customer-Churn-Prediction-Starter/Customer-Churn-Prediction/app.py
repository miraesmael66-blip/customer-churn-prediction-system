import os
from typing import Any

import requests
import streamlit as st

st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon="📊",
    layout="wide",
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
PREDICT_URL = f"{API_BASE_URL}/predict"
HEALTH_URL = f"{API_BASE_URL}/health"

st.markdown("""
<style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px;}
    .hero {padding: 1.4rem 1.6rem; border-radius: 16px; background: linear-gradient(120deg, #14213d, #243b63); color: white; margin-bottom: 1rem;}
    .muted {color: #64748b; font-size: 0.95rem;}
    div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.25); padding: 1rem; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="hero"><h1>Customer Churn Prediction System</h1>'
    '<p>Estimate whether a customer may leave, using the project’s trained machine-learning pipeline.</p></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("System status")
    try:
        health_response = requests.get(HEALTH_URL, timeout=3)
        if health_response.ok and health_response.json().get("model_loaded"):
            st.success("Backend connected · model ready")
        else:
            st.warning("Backend reachable, but model is not ready")
    except requests.RequestException:
        st.error("Backend unavailable")
    st.divider()
    st.caption("The interface sends inputs to FastAPI. The backend runs the saved model.")
    if st.button("Reset inputs"):
        st.session_state.clear()
        st.rerun()

st.subheader("Customer information")
st.caption("Enter values within the ranges observed in the training dataset. These are dataset ranges, not universal business limits.")

with st.form("customer_form"):
    section1, section2, section3 = st.columns(3)

    with section1:
        st.markdown("#### Usage")
        call_failure = st.number_input("Call Failure", min_value=0, max_value=36, value=8, step=1)
        seconds_of_use = st.number_input("Seconds of Use", min_value=0, max_value=17090, value=4370, step=1)
        frequency_of_use = st.number_input("Frequency of use", min_value=0, max_value=255, value=71, step=1)
        frequency_of_sms = st.number_input("Frequency of SMS", min_value=0, max_value=522, value=5, step=1)
        distinct_called_numbers = st.number_input("Distinct Called Numbers", min_value=0, max_value=97, value=17, step=1)

    with section2:
        st.markdown("#### Subscription")
        complains = st.selectbox("Complains", options=[0, 1], format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")
        subscription_length = st.number_input("Subscription Length", min_value=3, max_value=47, value=38, step=1)
        charge_amount = st.selectbox("Charge Amount", options=list(range(0, 11)))
        tariff_plan = st.selectbox("Tariff Plan", options=[1, 2], format_func=lambda x: f"Plan {x}")
        age_group = st.selectbox("Age Group", options=[1, 2, 3, 4, 5])

    with section3:
        st.markdown("#### Customer profile")
        age = st.number_input("Age", min_value=15, max_value=55, value=30, step=1)
        customer_value = st.number_input("Customer Value", min_value=0.0, max_value=2165.28, value=197.64, step=1.0)

    submitted = st.form_submit_button("Predict churn", type="primary", use_container_width=True)

if submitted:
    payload: dict[str, Any] = {
        "Call  Failure": int(call_failure),
        "Complains": int(complains),
        "Subscription  Length": int(subscription_length),
        "Charge  Amount": int(charge_amount),
        "Seconds of Use": int(seconds_of_use),
        "Frequency of use": int(frequency_of_use),
        "Frequency of SMS": int(frequency_of_sms),
        "Distinct Called Numbers": int(distinct_called_numbers),
        "Age Group": int(age_group),
        "Tariff Plan": int(tariff_plan),
        "Age": int(age),
        "Customer Value": float(customer_value),
    }
    try:
        with st.spinner("Requesting prediction from the FastAPI backend..."):
            response = requests.post(PREDICT_URL, json=payload, timeout=10)

        if response.status_code == 422:
            st.error("The submitted values did not pass backend validation. Check the input values and try again.")
        elif not response.ok:
            try:
                detail = response.json().get("detail", "The backend returned an error.")
            except ValueError:
                detail = "The backend returned an unreadable error response."
            st.error(f"Prediction unavailable: {detail}")
        else:
            result = response.json()
            st.divider()
            st.subheader("Prediction result")
            if result.get("predicted_class") == 1:
                st.error("Predicted class: Churn")
            else:
                st.success("Predicted class: No Churn")

            if result.get("churn_probability") is not None:
                probability = float(result["churn_probability"])
                if 0.0 <= probability <= 1.0:
                    left, right = st.columns([1, 2])
                    with left:
                        st.metric("Estimated churn probability", f"{probability:.1%}")
                    with right:
                        st.progress(probability)
                    st.caption(result.get("probability_note") or "This is an estimate, not a guarantee.")
            else:
                st.info("The loaded model did not provide a churn probability.")
    except requests.Timeout:
        st.error("The backend request timed out. Please try again.")
    except requests.ConnectionError:
        st.error("Could not connect to FastAPI. Confirm that the backend is running and API_BASE_URL is correct.")
    except requests.RequestException:
        st.error("The request could not be completed. Please try again.")

with st.expander("How this system works"):
    st.write(
        "The form sends customer features to a FastAPI endpoint. The backend validates the input, "
        "calculates Avg_Usage_Duration using the same formula as the notebook, and passes the ordered "
        "features to the saved scikit-learn model. Streamlit does not load or run the model itself."
    )
    st.write("A prediction is a model estimate and should not be treated as a guaranteed customer outcome.")

st.divider()
st.caption("Machine Learning Capstone · Customer Churn Prediction System")
