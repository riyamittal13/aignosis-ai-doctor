🩺 Aignosis – Your AI Medical Assistant
*Clarity Before Crisis*

Aignosis is an AI-powered medical assistant that predicts possible diseases based on user-reported symptoms and provides relevant health precautions.


🚀 Features
- 🔍 Takes symptoms as natural language input
- 🤖 Predicts top 3 likely diseases using an AI model
- 🛡 Displays relevant health precautions for each disease
- 🧠 Built with Sentence Transformers and Scikit-learn
- 🌐 Deployed live with Streamlit Cloud


🧪 Example Input
```text
"Red itchy rash on elbows and knees", "Flaking skin", "Skin irritation"


🔬Sample Output

🧬 Top Predicted Diseases
• Psoriasis – 39.82% confidence
⚠ No specific precautions found for Psoriasis

• Fungal infection – 9.28% confidence
🛡 Precautions:
✅ Use antifungal cream
✅ Keep area dry
✅ Consult a doctor

• Drug reaction – 6.86% confidence
🛡 Precautions:
✅ Stop suspected drug
✅ Consult doctor
✅ Drink plenty of water


⚒Tech Stack

Frontend/UI: Streamlit

Model: Scikit-learn (Logistic Regression)

Embeddings: SentenceTransformers (all-MiniLM-L6-v2)

Deployment: Streamlit Community Cloud

Data: Custom medical dataset + precautions mapping


📂Project Structure

aignosis-ai-doctor/
│
├── app.py                    # Main Streamlit app
├── disease_prediction_model.pkl
├── label_encoder.pkl
├── precautions.csv           # Contains diseases and precautions
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml           # Streamlit config for app title/logo
├── aignosis_logo.png.jpg     # Logo (optional)
└── README.md


🧠How It Works

1. User enters symptoms as comma-separated natural language phrases.

2. App encodes symptoms using Sentence Transformers (MiniLM-L6-v2).

3. Predicts disease probabilities via trained model.

4. Displays top 3 diseases and their relevant precautions using cosine similarity search.


📡Live Demo

🌐Try the App Here


👩🏻‍💻Author

Riya Mittal

B.Tech in Computer Science, Al-ML

GitHub: @riyamittal13


📜License

This project is for educational purposes only. Not intended for real-world medical diagnosis. Always consult a healthcare professional.
