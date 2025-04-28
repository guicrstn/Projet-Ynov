# app.py

import streamlit as st
from transformers import pipeline

# Configuration de la page
st.set_page_config(page_title="Analyse de Sentiment 📊", page_icon="🌐", layout="centered")

# Chargement du modèle
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

classifier = load_model()

# Initialisation de l'historique et du chat dans la session
if "history" not in st.session_state:
    st.session_state.history = []
if "chat_data" not in st.session_state:
    st.session_state.chat_data = {"prenom": "", "nom": "", "reponses": [], "sentiment": "", "grade": "", "color": ""}

# Personnalisation avancée du style Streamlit
st.markdown("""
    <style>
    .main {
        background-image: url('https://images.unsplash.com/photo-1507525428034-b723cf961d3e?ixlib=rb-4.0.3&auto=format&fit=crop&w=1600&q=80');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        padding: 2rem;
        border-radius: 10px;
    }
    .block-container {
        padding-top: 2rem;
        background-color: rgba(255, 255, 255, 0.85);
        border-radius: 10px;
    }
    .grade-circle {
        display: inline-block;
        width: 40px;
        height: 40px;
        line-height: 40px;
        border-radius: 50%;
        text-align: center;
        font-weight: bold;
        font-size: 20px;
        color: white;
    }
    .card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        transition: transform 0.2s;
    }
    .card:hover {
        transform: scale(1.02);
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à", ["Analyse de sentiment", "Historique", "À propos"])

if page == "Analyse de sentiment":
    st.markdown("""
    <h1 style='text-align: center;'>🌍 Analyse de Sentiment avec Intelligence Artificielle 📊</h1>
    """, unsafe_allow_html=True)

    st.info("""
    Répondez aux questions pour analyser votre humeur générale ! 💬
    """)

    def interpret_sentiment(label):
        if '1 star' in label or '2 stars' in label:
            return "Mauvais", "C", "#FF0000"
        elif '3 stars' in label:
            return "Bien", "B", "#FFA500"
        elif '4 stars' in label or '5 stars' in label:
            return "Excellent", "A", "#00FF00"
        else:
            return "Indéterminé", "?", "#808080"

    # Formulaire utilisateur
    prenom = st.text_input("👤 Votre prénom :", value=st.session_state.chat_data["prenom"])
    nom = st.text_input("👤 Votre nom :", value=st.session_state.chat_data["nom"])

    if prenom and nom:
        st.session_state.chat_data["prenom"] = prenom
        st.session_state.chat_data["nom"] = nom

        questions = [
            "Comment vous sentez-vous aujourd'hui ?",
            "Quel événement vous a marqué récemment ?",
            "Quelle est votre principale source de motivation ?"
        ]

        reponses = st.session_state.chat_data["reponses"]

        if len(reponses) < 1:
            rep1 = st.text_input(f"Question 1 : {questions[0]}", key="q1")
            if rep1:
                st.session_state.chat_data["reponses"].append(rep1)
                st.rerun()

        if len(reponses) == 1:
            rep2 = st.text_input(f"Question 2 : {questions[1]}", key="q2")
            if rep2:
                st.session_state.chat_data["reponses"].append(rep2)
                st.rerun()

        if len(reponses) == 2:
            rep3 = st.text_input(f"Question 3 : {questions[2]}", key="q3")
            if rep3:
                st.session_state.chat_data["reponses"].append(rep3)
                st.rerun()

        if len(reponses) == 3 and not st.session_state.chat_data["sentiment"]:
            combined_text = " ".join(reponses)
            result = classifier(combined_text)[0]
            sentiment, grade, color = interpret_sentiment(result['label'])
            st.session_state.chat_data.update({"sentiment": sentiment, "grade": grade, "color": color})
            st.session_state.history.append(dict(st.session_state.chat_data))
            st.rerun()

        if st.session_state.chat_data["sentiment"]:
            st.success(f"**Sentiment global détecté :** :sparkles: {st.session_state.chat_data['sentiment']} :sparkles:")
            st.markdown(f"""
                <div class="grade-circle" style="background-color: {st.session_state.chat_data['color']};">
                    {st.session_state.chat_data['grade']}
                </div>
            """, unsafe_allow_html=True)

            if st.button("🔄 Faire un nouveau test"):
                st.session_state.chat_data = {"prenom": "", "nom": "", "reponses": [], "sentiment": "", "grade": "", "color": ""}
                st.rerun()

elif page == "Historique":
    st.markdown("# 📜 Historique des demandes")
    if st.session_state.history:
        for item in reversed(st.session_state.history):
            card_html = f"""
            <div class='card'>
                <p><strong>👤 Prénom :</strong> {item['prenom']}</p>
                <p><strong>👤 Nom :</strong> {item['nom']}</p>
            """
            for idx, rep in enumerate(item['reponses'], 1):
                card_html += f"<p><strong>📝 Réponse {idx} :</strong> {rep}</p>"
            card_html += f"""
                <p><strong>📈 Sentiment détecté :</strong> {item['sentiment']}</p>
                <div class="grade-circle" style="background-color: {item['color']};">
                    {item['grade']}
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
    else:
        st.info("Aucune analyse enregistrée pour le moment.")

elif page == "À propos":
    st.markdown("""
    # À propos de ce projet

    Ce projet a été réalisé dans le cadre du **YNOV Campus** 🎓.

    L'objectif est de montrer comment utiliser un modèle d'intelligence artificielle pour analyser le sentiment d'une phrase et le déployer via une interface web interactive.

    **Technologies utilisées :**
    - Python 🐍
    - Transformers de Hugging Face 🤗
    - Streamlit 🚀

    Merci de votre visite ! 👋
    """)

st.markdown("""
---
<div style='text-align: center;'>
    Créé dans le cadre du projet <strong>YNOV Campus</strong> 🎓<br>
    Merci pour votre visite 👋
</div>
""", unsafe_allow_html=True)
