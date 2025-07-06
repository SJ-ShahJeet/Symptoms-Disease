from fastapi import FastAPI
from pydantic import BaseModel
import torch, pickle, pandas as pd, re, nltk, json, os
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util
from collections import Counter
from typing import List
from uuid import uuid4

# --- NLTK Setup ---
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# --- App Init ---
app = FastAPI()
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
SESSION_DIR = "sessions"
os.makedirs(SESSION_DIR, exist_ok=True)

# --- Load Assets ---
def load_assets():
    with open("model/symptom_embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)
    with open("model/symptom_list.pkl", "rb") as f:
        symptoms = pickle.load(f)
    with open("model/symptom_confidence_scores.pkl", "rb") as f:
        confidence = pickle.load(f)
    with open("model/model_disease_frequency.pkl", "rb") as f:
        freq = pickle.load(f)
    df = pd.read_csv("data/Final_Augmented_dataset_Diseases_and_Symptoms.csv")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model, embeddings, symptoms, confidence, freq, df

model, symptom_embeddings, symptom_list, symptom_confidence, disease_frequency, df = load_assets()

# --- Utils ---
def clean_and_tokenize(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    tokens = word_tokenize(text)
    return [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]

def extract_symptoms(text, top_k=10, min_score=0.7):
    tokens = clean_and_tokenize(text)
    phrases = [" ".join(gram) for n in range(1, 4) for gram in ngrams(tokens, n)]
    seen, matches = set(), []

    for phrase in phrases:
        input_vec = model.encode(phrase, convert_to_tensor=True)
        scores = [(sym, float(util.cos_sim(input_vec, emb))) for sym, emb in symptom_embeddings.items()]
        best_sym, best_score = max(scores, key=lambda x: x[1])
        if best_score >= min_score and best_sym not in seen:
            seen.add(best_sym)
            matches.append(best_sym)

    return matches[:top_k]

def suggest_related_symptoms(confirmed, top_k=5):
    valid = [s for s in confirmed if s in df.columns]
    if not valid:
        return [{"symptom": f"(Not in dataset: {s})", "count": 0} for s in confirmed]

    subset = df[df[valid].sum(axis=1) > 0]
    counts = Counter({col: int(subset[col].sum()) for col in df.columns[1:] if col not in valid})
    return [{"symptom": s, "count": c} for s, c in counts.most_common(top_k)]

def rank_diseases(symptoms, top_n=5):
    results = []
    for disease, sym_scores in symptom_confidence.items():
        matched = [s for s in symptoms if s in sym_scores]
        if not matched:
            continue
        confidence = sum(sym_scores[s] for s in matched)
        overlap = len(matched) / len(symptoms)
        prevalence = disease_frequency.get(disease, 0)
        score = 0.3 * confidence + 0.1 * overlap + 0.6 * prevalence
        results.append({
            "disease": disease,
            "score": round(score, 4),
            "confidence": round(confidence, 4),
            "overlap": round(overlap, 4),
            "prevalence": round(prevalence, 4),
        })
    return sorted(results, key=lambda x: x["score"], reverse=True)[:top_n]

# --- Session Management ---
def get_session_path(session_id):
    return os.path.join(SESSION_DIR, f"{session_id}.json")

def load_confirmed_symptoms(session_id):
    path = get_session_path(session_id)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f).get("confirmed", [])
    return []

def update_session_symptoms(session_id, new_symptoms):
    confirmed = load_confirmed_symptoms(session_id)
    updated = list(set(confirmed + new_symptoms))
    with open(get_session_path(session_id), "w") as f:
        json.dump({"confirmed": updated}, f)
    return updated

# --- Input Schema ---
class SymptomInput(BaseModel):
    message: str = ""  # ✅ Make optional
    session_id: str = None
    top_n: int = 5

# --- Endpoints ---
@app.post("/extract")
def extract_route(payload: SymptomInput):
    new = extract_symptoms(payload.message)
    session_id = payload.session_id or str(uuid4())
    
    confirmed = update_session_symptoms(session_id, new)
    
    # 🔍 DEBUG
    path = get_session_path(session_id)
    print(f"✅ Extract route called")
    print(f"📝 Saving to session file: {path}")
    print(f"💾 Confirmed symptoms: {confirmed}")
    
    return {
        "session_id": session_id,
        "input": payload.message,
        "extracted_symptoms": new,
        "confirmed_symptoms": confirmed
    }

@app.post("/suggest")
def suggest_route(payload: SymptomInput):
    print("📩 /suggest called with session_id:", payload.session_id)
    confirmed = load_confirmed_symptoms(payload.session_id)
    print("🔎 Loaded confirmed symptoms:", confirmed)

    if not confirmed:
        return {
            "session_id": payload.session_id,
            "confirmed_symptoms": confirmed,
            "suggested_symptoms": []
        }

    suggestions = suggest_related_symptoms(confirmed)
    return {
        "session_id": payload.session_id,
        "confirmed_symptoms": confirmed,
        "suggested_symptoms": suggestions
    }

@app.post("/predict")
def predict_route(payload: SymptomInput):
    confirmed = load_confirmed_symptoms(payload.session_id)
    predictions = rank_diseases(confirmed, payload.top_n)
    return {
        "session_id": payload.session_id,
        "confirmed_symptoms": confirmed,
        "predicted_diseases": predictions
    }