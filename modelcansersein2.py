# Importation des bibliothèques nécessaires
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement du modèle
modele = joblib.load("modell_svm.pkl")

# Titre et présentation de l'application
st.title("Application de Prédiction - Cancer du Sein")
st.markdown("""
Bienvenue dans cette application interactive de prédiction du cancer du sein.
Elle utilise un modèle de classification pour déterminer si une tumeur est :
- **Bénigne** (non cancéreuse), ou
- **Maligne** (cancéreuse).

---

### Instructions
1. Ajustez les curseurs correspondant aux caractéristiques de la tumeur pour une prédiction individuelle, ou
2. Téléchargez un fichier CSV contenant plusieurs observations pour obtenir des prédictions groupées.

### Attention
Les résultats sont basés sur un modèle préentraîné et ne remplacent **en aucun cas** un avis médical.
""")

# Liste des caractéristiques avec leurs plages de valeurs typiques
plages_caracteristiques = {
   "texture_mean": (9.0, 39.0),
    "smoothness_mean": (0.05, 0.15),
    "compactness_mean": (0.02, 0.35),
    "concave points_mean": (0.0, 0.2),
    "symmetry_mean": (0.12, 0.3),
    "fractal_dimension_mean": (0.04, 0.1),
    "texture_se": (0.3, 4.9),
    "area_se": (6.0, 550.0),
    "smoothness_se": (0.002, 0.03),
    "compactness_se": (0.002, 0.15),
    "concavity_se": (0.0, 0.4),
    "concave points_se": (0.0, 0.05),
    "symmetry_se": (0.008, 0.08),
    "fractal_dimension_se": (0.001, 0.03),
    "texture_worst": (12.0, 50.0),
    "area_worst": (200.0, 2500.0),
    "smoothness_worst": (0.07, 0.22),
    "compactness_worst": (0.025, 1.1),
    "concavity_worst": (0.0, 1.25),
    "concave points_worst": (0.0, 0.4),
    "symmetry_worst": (0.15, 0.6),
    "fractal_dimension_worst": (0.05, 0.25),
}

# Section : Prédiction pour une observation unique
st.header("Prédiction pour une observation unique")

# Création des entrées avec des curseurs
saisie_utilisateur = {}
col1, col2 = st.columns(2)

for i, (caracteristique, (val_min, val_max)) in enumerate(plages_caracteristiques.items()):
    with col1 if i % 2 == 0 else col2:
        saisie_utilisateur[caracteristique] = st.slider(
            f"{caracteristique.replace('_', ' ').capitalize()}",
            min_value=val_min,
            max_value=val_max,
            value=(val_min + val_max) / 2
        )

# Convertir les données utilisateur en DataFrame
saisie_utilisateur_df = pd.DataFrame([saisie_utilisateur])

# Prédiction pour une observation unique
if st.button("Prédire"):
    prediction = modele.predict(saisie_utilisateur_df)
    resultat = "Maligne (cancéreuse)" if prediction[0] == 1 else "Bénigne (non cancéreuse)"
    st.write(f"### Résultat de la prédiction : **{resultat}**")

    # Afficher les probabilités sous forme de graphique
    probabilites = modele.predict_proba(saisie_utilisateur_df)[0]
    fig, ax = plt.subplots()
    ax.bar(["Bénigne", "Maligne"], probabilites, color=["green", "red"])
    ax.set_ylabel("Probabilité")
    st.pyplot(fig)

# Section : Prédiction pour un fichier CSV
st.header("Prédiction pour un fichier CSV")

# Téléchargement du fichier CSV
fichier_telecharge = st.file_uploader("Téléchargez un fichier CSV contenant les caractéristiques des tumeurs", type=["csv"])

if fichier_telecharge is not None:
    # Chargement et affichage des données
    donnees = pd.read_csv(fichier_telecharge)
    st.write("### Aperçu des données chargées :")
    st.dataframe(donnees.head())

    # Vérification de la compatibilité avec le modèle
    if all(caracteristique in donnees.columns for caracteristique in plages_caracteristiques.keys()):
        if st.button("Prédire pour le fichier CSV"):
            predictions = modele.predict(donnees[list(plages_caracteristiques.keys())])
            donnees["Prédiction"] = ["Maligne" if p == 1 else "Bénigne" for p in predictions]
            st.write("### Résultats des prédictions :")
            st.dataframe(donnees)

            # Visualisation de la répartition des prédictions
            st.subheader("Distribution des prédictions")
            fig, ax = plt.subplots()
            sns.countplot(x=donnees["Prédiction"], palette={"Bénigne": "green", "Maligne": "red"}, ax=ax)
            ax.set_title("Nombre de cas bénins vs malins")
            ax.set_ylabel("Nombre de cas")
            st.pyplot(fig)

            # Matrice de corrélation
            st.subheader("Matrice de Corrélation")
            correlation = donnees[list(plages_caracteristiques.keys())].corr()
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            ax.set_title("Corrélation entre les différentes caractéristiques")
            st.pyplot(fig)

            # Boxplot des caractéristiques
            st.subheader("Boxplot des caractéristiques")
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.boxplot(data=donnees[list(plages_caracteristiques.keys())], ax=ax)
            ax.set_title("Boxplot des caractéristiques")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            st.pyplot(fig)

    else:
        st.error("Le fichier CSV ne contient pas toutes les colonnes nécessaires.")
