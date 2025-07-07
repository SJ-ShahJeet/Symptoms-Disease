
# 🩺 Symptom-to-Disease Assistant (FastAPI + Streamlit)

A smart assistant that analyzes natural language symptom descriptions and predicts possible disease categories using semantic embeddings and frequency-based scoring. This project simulates a **triage-like experience** — helping prioritize potential diagnoses from vague or detailed symptom inputs.

---

## 🚀 Features

- 🧠 **Semantic Symptom Extraction** from free-text (e.g., "I feel tightness in my chest and dizzy")
- 🤝 **Session Management** with confirmation and suggestions for related symptoms
- 📊 **Disease Ranking** using:
  - Symptom match confidence
  - Overlap ratio
  - Disease prevalence
- 🔌 Modular **FastAPI backend** with `/extract`, `/suggest`, and `/predict` endpoints
- 💡 User-friendly **Streamlit frontend** (if deployed)
- ✅ Code designed for **scalability** and **extensibility**

---

## 📁 Project Structure

```
.
├── app.py                      # Streamlit frontend
├── main.py                    # FastAPI backend logic
├── model/
│   ├── symptom_embeddings.pkl
│   ├── symptom_list.pkl
│   ├── symptom_confidence_scores.pkl
│   ├── model_disease_frequency.pkl
├── data/
│   └── Final_Augmented_dataset_Diseases_and_Symptoms.csv
├── notebooks/
│   └── render_model.ipynb     # Notebook for CPU-safe model conversion
├── render.yaml                # Render deployment spec (optional)
```

---

## 🔧 How It Works

1. **Symptom Extraction**  
   Uses `sentence-transformers` to semantically match phrases from input to known symptoms using cosine similarity.

2. **Symptom Suggestion**  
   Based on co-occurrence in the dataset, the system suggests related symptoms.

3. **Disease Prediction**  
   Ranks diseases by a composite score:  
   `score = 0.3 * confidence + 0.1 * overlap + 0.6 * prevalence`

---

## ⚙️ Technologies Used

- 🐍 Python 3.10+
- 🧬 `sentence-transformers`
- ⚡ FastAPI
- 🌐 Streamlit
- 📦 Pandas, NLTK
- 🧠 `pickle` for serialized model data
- ☁️ (Optional) Render for backend deployment

---

## 🧪 Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Start FastAPI backend
uvicorn main:app --reload

# (In a new terminal) Start Streamlit frontend
streamlit run app.py
```

---

## 📝 Notes

- Render deployment was attempted but deferred due to MPS-related compatibility issues with `.pkl` files.
- Code is modular and documented for clarity and future improvements.

---

## 📌 Why This Project?

This assistant was built as part of a real-world inspired project to simulate **early triage support**, showcasing:

- 🧠 NLP in healthcare
- 🔁 Session-based interaction design
- 💡 Intelligent rule-based scoring without ML overfitting
