# Importation des bibliothèques nécessaires
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement du modèle
model = joblib.load("modell_svm.pkl")

# Titre et présentation de l'application
st.title("Application de Prédiction - Cancer du Sein")
st.markdown("""
Bienvenue dans cette application interactive pour la prédiction du cancer du sein. 
Elle utilise un modèle de classification pour déterminer si une tumeur est :
- **Bénigne** (non cancéreuse), ou
- **Maligne** (cancéreuse).

---

### Instructions
1. Remplissez les champs correspondant aux caractéristiques de la tumeur pour une prédiction individuelle, ou
2. Téléchargez un fichier CSV contenant plusieurs observations pour obtenir des prédictions groupées.

### Attention
Les résultats sont basés sur un modèle préentraîné et ne remplacent **en aucun cas** un avis médical.
""")

# Section : Prédiction pour une observation unique
st.header("Prédiction pour une observation unique")

# Liste des caractéristiques (ajustez selon votre dataset)
features = [
    "texture_mean", "smoothness_mean", "compactness_mean", "concave points_mean", "symmetry_mean",
    "fractal_dimension_mean", "texture_se", "area_se", "smoothness_se", "compactness_se",
    "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se", "texture_worst",
    "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst",
    "symmetry_worst", "fractal_dimension_worst"
]

# Créer des champs pour chaque caractéristique
user_input = {}
col1, col2 = st.columns(2)
for i, feature in enumerate(features):
    with col1 if i % 2 == 0 else col2:
        user_input[feature] = st.number_input(f"{feature.capitalize()}", value=0.0)

# Convertir les données utilisateur en DataFrame
user_input_df = pd.DataFrame([user_input])

# Prédiction pour une observation unique
if st.button("Prédire"):
    prediction = model.predict(user_input_df)
    result = "Maligne (cancéreuse)" if prediction[0] == 1 else "Bénigne (non cancéreuse)"
    st.write(f"### Résultat de la prédiction : **{result}**")

    # Afficher les probabilités sous forme de graphique
    probabilities = model.predict_proba(user_input_df)[0]
    fig, ax = plt.subplots()
    ax.bar(["Bénigne", "Maligne"], probabilities, color=["green", "red"])
    ax.set_ylabel("Probabilité")
    st.pyplot(fig)

# Section : Prédiction pour un fichier CSV
st.header("Prédiction pour un fichier CSV")

# Téléchargement du fichier CSV
uploaded_file = st.file_uploader("Téléchargez un fichier CSV contenant les caractéristiques des tumeurs", type=["csv"])

if uploaded_file is not None:
    # Chargement et affichage des données
    data = pd.read_csv(uploaded_file)
    st.write("### Aperçu des données chargées :")
    st.dataframe(data.head())

    # Vérification de la compatibilité avec le modèle
    if all(feature in data.columns for feature in features):
        if st.button("Prédire pour le fichier CSV"):
            predictions = model.predict(data[features])
            data["Prédiction"] = ["Maligne" if p == 1 else "Bénigne" for p in predictions]
            st.write("### Résultats des prédictions :")
            st.dataframe(data)

            # Visualisation de la répartition des prédictions
            st.subheader("Distribution des prédictions")
            fig, ax = plt.subplots()
            sns.countplot(x=data["Prédiction"], palette={"Bénigne": "green", "Maligne": "red"}, ax=ax)
            ax.set_title("Nombre de cas bénins vs malins")
            ax.set_ylabel("Nombre de cas")
            st.pyplot(fig)

    else:
        st.error("Le fichier CSV ne contient pas toutes les colonnes nécessaires.")
