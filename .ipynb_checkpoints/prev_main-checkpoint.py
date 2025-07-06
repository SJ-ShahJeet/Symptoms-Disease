# main.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
import torch
import pickle
from sentence_transformers import SentenceTransformer, util
import re
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# --- Setup ---
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = FastAPI()

# --- Load Model and Embeddings ---
model = SentenceTransformer("all-MiniLM-L6-v2")
with open("model/symptom_embeddings.pkl", "rb") as f:
    symptom_embeddings = pickle.load(f)
with open("model/symptom_list.pkl", "rb") as f:
    symptom_list = pickle.load(f)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# --- Helper Functions ---
def clean_and_tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens

def extract_symptoms_semantic_phrases(text, top_k=5):
    tokens = clean_and_tokenize(text)
    phrases = []
    for n in range(1, 4):
        phrases += [" ".join(gram) for gram in ngrams(tokens, n)]

    seen = set()
    matches = []
    for phrase in phrases:
        input_vec = model.encode(phrase, convert_to_tensor=True)
        scored = [(sym, float(util.cos_sim(input_vec, emb))) for sym, emb in symptom_embeddings.items()]
        best_symptom, best_score = max(scored, key=lambda x: x[1])
        if best_score > 0.6 and best_symptom not in seen:
            seen.add(best_symptom)
            matches.append((best_symptom, best_score))
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches[:top_k]

# --- API Input Model ---
class SymptomInput(BaseModel):
    message: str

# --- Route ---
@app.post("/symptoms")
def extract_symptoms_endpoint(payload: SymptomInput):
    symptoms = extract_symptoms_semantic_phrases(payload.message)
    return {"extracted_symptoms": symptoms}