# app.py

import streamlit as st
import requests
import uuid

# --- Config ---
BASE_URL = "http://localhost:8000"
st.set_page_config(page_title="Predictive Triage Assistant", layout="centered")
st.title("🩺 Predictive Triage Assistant")

# --- Session ID ---
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())
session_id = st.session_state["session_id"]

# --- User Input ---
st.subheader("Enter your symptoms")
message = st.text_area("Describe your symptoms in natural language")

# --- Extract + Suggest ---
if st.button("🧠 Extract Symptoms"):
    if not message.strip():
        st.warning("Please describe what you're feeling.")
    else:
        with st.spinner("Analyzing symptoms..."):
            res = requests.post(f"{BASE_URL}/extract", json={"message": message, "session_id": session_id})
            if res.status_code == 200:
                data = res.json()
                extracted = data.get("extracted_symptoms", [])
                if extracted:
                    st.success("✅ New Symptoms Extracted:")
                    for sym in extracted:
                        st.markdown(f"- {sym}")
                else:
                    st.info("No new symptoms found.")
            else:
                st.error("Symptom extraction failed.")

# --- Confirmed Symptoms ---
fetch = requests.post(f"{BASE_URL}/extract", json={"message": "", "session_id": session_id})
if fetch.status_code == 200:
    confirmed = fetch.json().get("confirmed_symptoms", [])
    if confirmed:
        st.markdown("### ✅ Confirmed Symptoms So Far:")
        checked = []
        for sym in confirmed:
            if st.checkbox(sym, value=True, key=f"confirmed_{sym}"):
                checked.append(sym)
        st.session_state["selected_symptoms"] = checked
    else:
        st.info("ℹ️ No confirmed symptoms yet.")
else:
    st.error("Unable to fetch confirmed symptoms.")

# --- Suggest Co-occurring with Checkboxes ---
suggest_res = requests.post(f"{BASE_URL}/suggest", json={"session_id": session_id})
if suggest_res.status_code == 200:
    suggestions = suggest_res.json().get("suggested_symptoms", [])
    if suggestions:
        st.markdown("💡 **Do any of these apply to you?**")
        selected_suggestions = []
        for item in suggestions:
            label = item["symptom"]
            if st.checkbox(label, key=f"suggestion_{label}"):
                selected_suggestions.append(label)

        if selected_suggestions:
            if st.button("➕ Add Selected Symptoms"):
                update = requests.post(f"{BASE_URL}/extract", json={
                    "message": " ".join(selected_suggestions),
                    "session_id": session_id
                })
                if update.status_code == 200:
                    st.success("✅ Selected symptoms added!")
                    st.rerun()
    else:
        st.info("No related symptoms found.")
else:
    st.warning("Could not fetch co-occurring suggestions.")

# --- Prediction ---
if st.button("🔮 Predict Diseases"):
    with st.spinner("Predicting..."):
        res = requests.post(f"{BASE_URL}/predict", json={
            "session_id": session_id,
            "top_n": 5
        })
        if res.status_code == 200:
            result = res.json()
            diseases = result.get("predicted_diseases", [])
            if diseases:
                st.markdown("### 🧬 Possible Conditions:")
                for d in diseases:
                    st.markdown(
                        f"**{d['disease'].title()}** — Score: `{d['score']}`  \n"
                        f"*Confidence:* `{d['confidence']}`, "
                        f"*Overlap:* `{d['overlap']}`, "
                        f"*Prevalence:* `{d['prevalence']}`"
                    )
            else:
                st.info("No diseases matched your symptoms.")
        else:
            st.error("Prediction failed.")

# --- Reset Session ---
st.markdown("---")
if st.button("🔄 Reset Session"):
    st.session_state.clear()
    st.rerun()