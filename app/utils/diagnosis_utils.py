import re
import pickle
from nltk.corpus import stopwords
from sentence_transformers import util, SentenceTransformer

# Load models
with open("model/model_symptom_embeddings.pkl", "rb") as f:
    symptom_embeddings = pickle.load(f)

with open("model/symptom_confidence_scores.pkl", "rb") as f:
    symptom_confidence = pickle.load(f)

with open("model/model_disease_frequency.pkl", "rb") as f:
    disease_frequency = pickle.load(f)

symptom_list = list(symptom_embeddings.keys())
stop_words = set(stopwords.words("english"))
model = SentenceTransformer("all-MiniLM-L6-v2")

# Core Functions
def clean_input(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    return " ".join([w for w in words if w not in stop_words])

def extract_symptoms(text, top_k=10):
    cleaned = clean_input(text)
    input_vec = model.encode(cleaned, convert_to_tensor=True).cpu()
    scores = [(sym, float(util.cos_sim(input_vec, vec))) for sym, vec in symptom_embeddings.items()]
    return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

def rank_diseases(symptoms, top_n=5):
    results = []
    for disease, sym_scores in symptom_confidence.items():
        matched = [s for s in symptoms if s in sym_scores]
        if not matched:
            continue
        confidence = sum(sym_scores[s] for s in matched)
        overlap = len(matched) / len(symptoms)
        prevalence = disease_frequency.get(disease, 0)
        score = 0.45 * confidence + 0.15 * overlap + 0.4 * prevalence
        results.append({
            "disease": disease,
            "score": round(score, 4),
            "confidence": round(confidence, 4),
            "overlap": round(overlap, 4),
            "prevalence": round(prevalence, 4)
        })
    return sorted(results, key=lambda x: x["score"], reverse=True)[:top_n]