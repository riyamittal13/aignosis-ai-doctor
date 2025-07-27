from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
import streamlit as st
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

#Loading precautions file
precautions_df = pd.read_csv('precautions.csv')

# Load the models
clf = joblib.load("disease_prediction_model.pkl")
le = joblib.load("label_encoder.pkl")

import torch
model_st = SentenceTransformer('all-MiniLM-L6-v2')
if torch.cuda.is_available():
    model_st = model_st.to('cuda')
else:
    model_st = model_st.to('cpu')

# Precompute embeddings of disease names from precautions.csv
precaution_diseases = precautions_df['Disease'].tolist()
precaution_embeddings = model_st.encode(precaution_diseases)

# Predict disease from symptoms
def predict_disease_from_symptoms(symptom_list):
    embeddings = model_st.encode(symptom_list)
    avg_embedding = np.mean(embeddings, axis=0).reshape(1, -1)
    probabilities = clf.predict_proba(avg_embedding)[0]
    top_indices = np.argsort(probabilities)[::-1][:3]
    top_diseases = le.inverse_transform(top_indices)
    top_probs = probabilities[top_indices]
    return [(disease, round(prob * 100, 2)) for disease, prob in zip(top_diseases, top_probs)]

# UI
st.set_page_config(page_title="Aignosis - Your AI Doctor", layout="centered")

st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 10px;
    }
    .stTextArea textarea {
        border-radius: 10px;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

from PIL import Image
logo = Image.open("aignosis_logo.png.jpg")  # Make sure this filename matches your image
st.image(logo, width=200, caption="Aignosis – Clarity Before Crisis")

st.title("🩺 Aignosis")
st.subheader("Clarity Before Crisis")

st.markdown("""
Enter your *symptoms separated by commas* (e.g., fever, headache, fatigue)  
Aignosis will analyze and suggest possible diseases with precautions.
""")

user_input = st.text_area("📝 Describe your symptoms here:", placeholder="e.g. fever, cough, nausea", height=150)

if st.button("🔍 Diagnose"):
    if user_input.strip() == "":
        st.warning("Please type your symptoms to get results.")
    else:
        symptoms = [s.strip() for s in user_input.split(",")]
        predictions = predict_disease_from_symptoms(symptoms)
        
        from sklearn.metrics.pairwise import cosine_similarity
        
        st.markdown("### 🧬 Top Predicted Diseases")
        for i, (disease, prob) in enumerate(predictions, start=1):
            st.markdown(f"{i}. {disease}** — {prob}% confidence")
            
            disease_embedding = model_st.encode([disease])[0]
            similarities = cosine_similarity([disease_embedding], precaution_embeddings)[0]
            best_match_index = similarities.argmax()
            best_match_score = similarities[best_match_index]
            best_match_disease = precaution_diseases[best_match_index]
            
            precaution_row = precautions_df[precautions_df["Disease"].str.lower() == best_match_disease.lower()]
            if best_match_score >= 0.7 and not precaution_row.empty:
                st.markdown(f"#### 🛡 Precautions for *{best_match_disease}*")
                for i in range(1, 4):
                    precaution = precaution_row[f"Precaution_{i}"].values[0]
                    st.markdown(f"- ✅ {precaution}")
            else:
                st.warning(f"⚠ No specific precautions found for *{disease}*.")

st.markdown("---")
st.caption("Made with ❤ using Streamlit and Sentence Transformers · Aignosis · 2025")