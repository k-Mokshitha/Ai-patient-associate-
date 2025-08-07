import streamlit as st
import requests
import os
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

# Load API key and deployment URL from .env
load_dotenv()
API_KEY = os.getenv("IBM_API_KEY")
DEPLOYMENT_URL = os.getenv("DEPLOYMENT_URL")

# Get IBM IAM token
def get_token():
    url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = f"apikey={API_KEY}&grant_type=urn:ibm:params:oauth:grant-type:apikey"
    response = requests.post(url, headers=headers, data=data)
    return response.json()['access_token']

# Send prompt to IBM Granite model
def ask_ibm(prompt):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {get_token()}"
    }
    data = {
        "input": prompt,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 300
        }
    }
    response = requests.post(f"{DEPLOYMENT_URL}/v2/text/generate", headers=headers, json=data)
    return response.json()['results'][0]['generated_text']

# Streamlit UI
st.set_page_config(page_title="AI Medical Assistant", layout="wide")
st.title("🤖 AI Medical Assistant")

tab1, tab2, tab3, tab4 = st.tabs([
    "🗨️ Patient Chat",
    "🩺 Disease Prediction",
    "💊 Treatment Plan",
    "📊 Health Analytics"
])

# Tab 1: Patient Chat
with tab1:
    st.subheader("Chat with the AI Assistant")
    prompt = st.text_area("Ask a health-related question:")
    if st.button("Ask"):
        if prompt:
            with st.spinner("Thinking..."):
                answer = ask_ibm(prompt)
                st.success(answer)

# Tab 2: Disease Prediction (Simulated)
with tab2:
    st.subheader("Disease Prediction (Simulated)")
    symptoms = st.text_input("Enter symptoms (comma-separated):")
    if st.button("Predict Disease"):
        if symptoms:
            st.warning("This is a simulated result.")
            st.success("Predicted Disease: Flu")

# Tab 3: AI Treatment Plan
with tab3:
    st.subheader("Get a Treatment Plan from AI")
    diagnosis = st.text_input("Enter disease name:")
    if st.button("Generate Plan"):
        if diagnosis:
            plan_prompt = f"Suggest a simple and safe treatment plan for {diagnosis}."
            with st.spinner("Generating..."):
                treatment = ask_ibm(plan_prompt)
                st.success(treatment)

# Tab 4: Health Analytics
with tab4:
    st.subheader("Patient Health Analytics")
    try:
        df = pd.read_csv("data/patient_data.csv")
        st.write("### Patient Data", df)
        fig = px.line(df, x="Date", y="BloodPressure", title="📈 Blood Pressure Over Time")
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error loading data: {e}")
