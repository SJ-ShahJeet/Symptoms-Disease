import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from collections import defaultdict, Counter
from operator import itemgetter

# Load data
df = pd.read_csv("data/Final_Augmented_dataset_Diseases_and_Symptoms.csv")
all_symptoms = df.columns.tolist()[1:]

# Build maps
disease_symptom_map = {}
for disease in df['diseases'].unique():
    rows = df[df['diseases'] == disease]
    symptoms = set()
    for _, row in rows.iterrows():
        symptoms.update([s for s in all_symptoms if row[s] == 1])
    disease_symptom_map[disease] = list(symptoms)

symptom_disease_map = defaultdict(list)
for disease, symptoms in disease_symptom_map.items():
    for s in symptoms:
        symptom_disease_map[s].append(disease)

# Load model and embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
symptom_embeddings = model.encode(all_symptoms, convert_to_tensor=True)

# Session state
if 'confirmed_symptoms' not in st.session_state:
    st.session_state.confirmed_symptoms = []

# Title
st.title("🩺 Symptom-to-Diagnosis Assistant")
st.markdown("Hello! Tell me how you’re feeling and I’ll try to help suggest possible causes.")

# Input
user_input = st.text_input("👤 You:", "")

# Symptom matching
def extract_symptoms(user_text, top_n=5):
    emb = model.encode(user_text, convert_to_tensor=True)
    scores = util.cos_sim(emb, symptom_embeddings)[0]
    matched = sorted(zip(all_symptoms, scores), key=lambda x: float(x[1]), reverse=True)
    return [sym for sym, _ in matched[:top_n]]

# Related symptom suggestions
def suggest_related(confirmed, max_suggestions=5):
    related = Counter()
    for sym in confirmed:
        for dis in symptom_disease_map.get(sym, []):
            for s in disease_symptom_map.get(dis, []):
                if s not in confirmed:
                    related[s] += 1
    return [s for s, _ in related.most_common(max_suggestions)]

# Disease ranking
def rank_diseases(symptoms, top_n=5):
    score = {}
    for dis, sym_set in disease_symptom_map.items():
        match = len(set(symptoms) & set(sym_set))
        if match > 0:
            score[dis] = match / len(sym_set)
    ranked = sorted(score.items(), key=itemgetter(1), reverse=True)
    return [(d, round(s * 100, 2)) for d, s in ranked[:top_n]]

# App logic
if user_input:
    matched = extract_symptoms(user_input)
    st.session_state.confirmed_symptoms.extend([m for m in matched if m not in st.session_state.confirmed_symptoms])

    st.markdown(f"**🤖 Matched symptoms:** {', '.join(matched)}")
    if st.session_state.confirmed_symptoms:
        st.markdown(f"**✅ Confirmed symptoms so far:** {', '.join(st.session_state.confirmed_symptoms)}")
        related = suggest_related(st.session_state.confirmed_symptoms)
        if related:
            st.markdown(f"**💡 Do you also feel any of these?** {', '.join(related)}")
        else:
            st.markdown("👍 Thanks! That’s enough to make a guess.")
            predictions = rank_diseases(st.session_state.confirmed_symptoms)
            st.markdown("### 🧾 Possible Diagnoses:")
            for disease, confidence in predictions:
                st.write(f"• {disease} — {confidence:.2f}% match")
