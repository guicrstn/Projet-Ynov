# Importation des bibliothèques
import streamlit as st
from transformers import pipeline

# Configuration générale de la page Streamlit
st.set_page_config(page_title="Analyse de Sentiment 📊", page_icon="🌐", layout="centered")

# Chargement du modèle IA de Hugging Face
@st.cache_resource  # Cache pour éviter de recharger le modèle à chaque rafraîchissement
def load_model():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

classifier = load_model()

# Initialisation de l'historique et des données utilisateur dans la session
if "history" not in st.session_state:
    st.session_state.history = []  # Stocke toutes les analyses faites
if "chat_data" not in st.session_state:
    st.session_state.chat_data = {
        "prenom": "", "nom": "", "reponses": [], "sentiment": "", "grade": "", "color": ""
    }  # Stocke les réponses du test en cours

# Personnalisation de l'apparence du site avec du CSS
st.markdown("""
    <style>
    .main {
        background-image: url('...');  /* Image de fond */
        background-size: cover;
        ...
    }
    .block-container {
        padding-top: 2rem;
        background-color: rgba(255, 255, 255, 0.85);
        border-radius: 10px;
    }
    .grade-circle {
        /* Style du rond de note (A, B, C) */
    }
    .card {
        /* Style des cartes dans l'historique */
    }
    .card:hover {
        transform: scale(1.02); /* Animation au survol */
    }
    </style>
""", unsafe_allow_html=True)

# Barre latérale de navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à", ["Analyse de sentiment", "Historique", "À propos"])

# --- Page principale : Analyse de Sentiment ---
if page == "Analyse de sentiment":
    st.markdown("<h1 style='text-align: center;'>🌍 Analyse de Sentiment avec Intelligence Artificielle 📊</h1>", unsafe_allow_html=True)
    st.info("Répondez aux questions pour analyser votre humeur générale ! 💬")

    # Fonction pour interpréter le résultat IA
    def interpret_sentiment(label):
        if '1 star' in label or '2 stars' in label:
            return "Mauvais", "C", "#FF0000"
        elif '3 stars' in label:
            return "Bien", "B", "#FFA500"
        elif '4 stars' in label or '5 stars' in label:
            return "Excellent", "A", "#00FF00"
        else:
            return "Indéterminé", "?", "#808080"

    # Formulaire utilisateur : Prénom et Nom
    prenom = st.text_input("👤 Votre prénom :", value=st.session_state.chat_data["prenom"])
    nom = st.text_input("👤 Votre nom :", value=st.session_state.chat_data["nom"])

    if prenom and nom:
        # Mise à jour du prénom et nom dans la session
        st.session_state.chat_data["prenom"] = prenom
        st.session_state.chat_data["nom"] = nom

        # Questions posées à l'utilisateur
        questions = [
            "Comment vous sentez-vous aujourd'hui ?",
            "Quel événement vous a marqué récemment ?",
            "Quelle est votre principale source de motivation ?"
        ]

        reponses = st.session_state.chat_data["reponses"]

        # Gestion de l'affichage des 3 questions une par une
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

        # Après les 3 réponses ➔ analyse de sentiment
        if len(reponses) == 3 and not st.session_state.chat_data["sentiment"]:
            combined_text = " ".join(reponses)  # Regroupe les réponses
            result = classifier(combined_text)[0]  # Analyse IA
            sentiment, grade, color = interpret_sentiment(result['label'])  # Interprétation
            st.session_state.chat_data.update({
                "sentiment": sentiment, "grade": grade, "color": color
            })
            st.session_state.history.append(dict(st.session_state.chat_data))  # Sauvegarde dans l'historique
            st.rerun()

        # Affichage du résultat
        if st.session_state.chat_data["sentiment"]:
            st.success(f"**Sentiment global détecté :** :sparkles: {st.session_state.chat_data['sentiment']} :sparkles:")

            # Affichage du grade (A, B ou C)
            st.markdown(f"""
                <div class="grade-circle" style="background-color: {st.session_state.chat_data['color']};">
                    {st.session_state.chat_data['grade']}
                </div>
            """, unsafe_allow_html=True)

            # Bouton pour relancer un nouveau test
            if st.button("🔄 Faire un nouveau test"):
                st.session_state.chat_data = {"prenom": "", "nom": "", "reponses": [], "sentiment": "", "grade": "", "color": ""}
                st.rerun()

# --- Page Historique ---
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

# --- Page À propos ---
elif page == "À propos":
    st.markdown("""
    # À propos de ce projet

    Ce projet a été réalisé dans le cadre du **YNOV Campus** 🎓.

    **Technologies utilisées :**
    - Python 🐍
    - Transformers de Hugging Face 🤗
    - Streamlit 🚀

    Merci de votre visite ! 👋
    """)

# --- Footer ---
st.markdown("""
---
<div style='text-align: center;'>
    Créé dans le cadre du projet <strong>YNOV Campus</strong> 🎓<br>
    Merci pour votre visite 👋
</div>
""", unsafe_allow_html=True)
