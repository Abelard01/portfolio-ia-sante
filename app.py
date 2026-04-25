import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from PIL import Image
from utils import AsthmaXGB
import os


st.set_page_config(page_title="Portfolio | Abelard Mugisha", page_icon="🧬", layout="wide")
# --- INJECTION DU FOND ANIMÉ (THÈME CLAIR PREMIUM) ---
fond_anime_clair = """
<style>
@keyframes gradientAnimation {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

[data-testid="stAppViewContainer"] {
    /* Dégradé subtil de blancs et de gris très clairs */
    background: linear-gradient(-45deg, #ffffff, #f0f2f6, #e6eaf1, #f8f9fa);
    background-size: 400% 400%;
    animation: gradientAnimation 15s ease infinite;
}

[data-testid="stSidebar"] {
    /* Barre latérale blanche avec effet verre dépoli (glassmorphism) */
    background-color: rgba(255, 255, 255, 0.7) !important;
    backdrop-filter: blur(10px);
}
</style>
"""
st.markdown(fond_anime_clair, unsafe_allow_html=True)
# --------------------------------------------------------
# 2. Le Header
col_photo, col_texte = st.columns([1, 3])

with col_photo:
    # On remplace le st.info par le chargement de la vraie image
    # 3. Chargement de l'image avec diagnostic complet
    try:
        image_photo = Image.open("images/maphoto.JPG") 
        st.image(image_photo, use_container_width=True)
        
        # LA LIGNE MAGIQUE POUR AFFICHER L'IMAGE :
        st.image(image_photo, use_container_width=True) 
        
    except Exception as e:
        st.error(f"Erreur Image : {e}")
with col_texte:
    st.title("Abelard Mugisha")
    st.subheader("Ingénieur Biomédical & Data Scientist Santé")
    st.write("""
    Passionné par l'Intelligence Artificielle appliquée au monde de la santé. 
    Je conçois des modèles de Machine Learning et développe des architectures MLOps 
    pour transformer la donnée médicale en solutions cliniques fiables.
    """)
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 3])
    with col_btn1:
        st.link_button(" Mon LinkedIn", "https://www.linkedin.com/in/abelard-mugisha", use_container_width=True)
    with col_btn2:
        st.link_button(" Mon GitHub", "https://github.com/Abelard01", use_container_width=True)

st.divider()

st.header("Mes Projets Interactifs")
st.write("Sélectionnez un projet dans le menu de gauche pour le tester en direct.")

# 3. Le Menu de Navigation à gauche (Sidebar)
st.sidebar.header(" Mes Projets")
# Les choix avec leurs émojis intégrés
projet_choisi = st.sidebar.radio(
    "Choisissez un projet à tester :",
    [" DiagnosHand (Vision)", " IA Prédictive (Asthme)"]
)

st.sidebar.divider()
st.sidebar.info(" Sélectionnez un projet au-dessus pour voir l'interface changer en temps réel.")

# --- 4. AFFICHAGE CONDITIONNEL SELON LE PROJET CHOISI ---

if projet_choisi == " DiagnosHand (Vision)":
    st.subheader(" DiagnosHand : Triage IA (SOS Main)")
    st.write("Déploiement d'un  API pour assister les services d'urgence  SOS Main dans le pré-triage des patients. À partir d'une simple photographie de la plaie, le modèle de Deep Learning prédit la nécessité d'une intervention :Suture simple ou exploration chirurgicale (cas grave nécessitant un bloc).")


    
    # Note de transparence très appréciée par les recruteurs techniques
    st.info("**Note MLOps** : L'inférence réelle nécessitant une puissance de calcul GPU , cette interface présente la maquette front-end (Mock API) du dispositif.")
    
    # 1. Importateur d'image
    uploaded_file = st.file_uploader("Transférez une radiographie ou photo de la lésion (JPG, PNG)", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        from PIL import Image
        import time
        import random
        
        # 2. Lecture de l'image
        image = Image.open(uploaded_file)
        
        col_img, col_res = st.columns(2)
        
        with col_img:
            st.image(image, caption="Image médicale transférée", use_container_width=True)
            
        with col_res:
            st.write("### Contrôle de l'analyse")
            
            if st.button(" Lancer le diagnostic IA", use_container_width=True):
                
                # Simulation du temps de calcul réseau et GPU
                with st.spinner("Transmission au serveur GPU et analyse par le réseau de neurones en cours..."):
                    time.sleep(2.5) 
                
                # Simulation de prédictions réalistes
                confiance = random.uniform(88.5, 98.2)
                risques_possibles = [
                    ("Élevé (Suspicion de fracture / Lésion profonde)", ),
                    ("Modéré (Entorse sévère / Déchirure)", ),
                    ("Faible (Lésion superficielle)", )
                ]
                diagnostic = random.choice(risques_possibles)
                
                st.success("Inférence terminée avec succès.")
                st.divider()
                
                # 3. Le Rapport Clinique
                st.markdown("###  Rapport Clinique Préliminaire")
                
                if "Élevé" in diagnostic:
                    st.error(f"**Urgence détectée : {diagnostic}**")
                    st.write(" **Recommandation :** Orientation immédiate vers un chirurgien de la main.")
                elif "Modéré" in diagnostic:
                    st.warning(f"**Attention : {diagnostic}**")
                    st.write(" **Recommandation :** Consultation médicale recommandée sous 24/48h.")
                else:
                    st.success(f" **Diagnostic : {diagnostic}**")
                    st.write(" **Recommandation :** Soins de premiers secours classiques.")
                
                st.metric(label="Indice de confiance du modèle", value=f"{confiance:.1f}%")
                
                # 4. L'effet "Waouh" technique pour le recruteur
                with st.expander(" Voir les détails de l'architecture modèle"):
                    st.write("**Architecture** : BiomedCLIP Multimodal")
                    st.write("**Prétraitement** : Redimensionnement (224x224), Normalisation des pixels, Data Augmentation")
                    st.write("**Temps d'inférence (simulé)** : 2.45 secondes")
                    st.write("**Fonction de perte** : Categorical Cross-Entropy")

elif projet_choisi == " IA Prédictive (Asthme)":
    st.subheader(" IA Prédictive : Exacerbations de l'Asthme (Real-World Data)")
    st.write("Modèles entraînés sur des données d'historique médical et de facturation. Le système effectue simultanément une classification (risque de crise) et une régression (sévérité).")
    
    tab_patient, tab_synth = st.tabs([" 1. Dossier Patient & Prédiction", " 2. Données Synthétiques"])
    
    # --- ONGLET 1 : DOSSIER PATIENT (CLASSIFICATION + RÉGRESSION) ---
    with tab_patient:
        with st.form("asthma_master_form"):
            st.markdown("** Démographie & Suivi**")
            col1, col2, col3 = st.columns(3)
            with col1:
                index_age = st.number_input("Âge (index_age)", 18, 100, 45)
                female = st.selectbox("Sexe", [0, 1], format_func=lambda x: "Femme (1)" if x == 1 else "Homme (0)")
            with col2:
                total_cannisters = st.number_input("Inhalateurs utilisés (1 an)", 0, 50, 5)
                adherence = st.slider("Score d'adhésion au traitement", 0.0, 1.0, 0.8)
            with col3:
                pre_asthma_days = st.number_input("Jours liés à l'asthme", 0, 365, 10)
                previous_drugs = st.number_input("Nbr médicaments pris", 0, 20, 2)
                drug_s = st.selectbox("Indicateur traitement (drug_s)", [0, 1])
                
            st.markdown("** Historique Médical (Comorbidités)**")
            # Un multiselect est beaucoup plus élégant que 7 cases à cocher séparées
            comorbidites = st.multiselect(
                "Sélectionnez les antécédents du patient :",
                ["pneumonia", "sinusitis", "acute_bronchitis", "acute_laryngitis", "upper_respiratory_infection", "gerd", "rhinitis"],
                default=["sinusitis", "gerd"]
            )
            
            st.markdown("** Données Financières (Pré-Index)**")
            col4, col5, col6 = st.columns(3)
            with col4:
                total_charge = st.number_input("Charges totales ($)", 0, 100000, 2500)
            with col5:
                asthma_charge = st.number_input("Charges asthme ($)", 0, 100000, 800)
            with col6:
                pharma_charge = st.number_input("Charges pharma ($)", 0, 100000, 300)

            submit_patient = st.form_submit_button(" Lancer l'IA (Classification & Régression)")
            
        if submit_patient:
            # 1. Reconstitution EXACTE des variables pour tes modèles .pkl
            input_data = {
                "index_age": index_age,
                "female": female,
                "pneumonia": 1 if "pneumonia" in comorbidites else 0,
                "sinusitis": 1 if "sinusitis" in comorbidites else 0,
                "acute_bronchitis": 1 if "acute_bronchitis" in comorbidites else 0,
                "acute_laryngitis": 1 if "acute_laryngitis" in comorbidites else 0,
                "upper_respiratory_infection": 1 if "upper_respiratory_infection" in comorbidites else 0,
                "gerd": 1 if "gerd" in comorbidites else 0,
                "rhinitis": 1 if "rhinitis" in comorbidites else 0,
                "previous_asthma_drugs": previous_drugs,
                "total_pre_index_cannisters_365": total_cannisters,
                "adherence": adherence,
                "pre_asthma_days": pre_asthma_days,
                "drug_s": drug_s,
                "total_pre_index_charge": total_charge,
                "pre_asthma_charge": asthma_charge,
                "pre_asthma_pharma_charge": pharma_charge,
                "log_charges": np.log1p(total_charge),
                "log_asthma_charge": np.log1p(asthma_charge)
            }
            
            # Transformation en tableau Pandas
            df_patient = pd.DataFrame([input_data])
            
            # --- LA VRAIE CONNEXION AUX MODÈLES ---
            try:
                modele_classif = joblib.load("models/modele_classification.pkl")
                modele_reg = joblib.load("models/modele_regression.pkl")
                
                # 2. Lancement des prédictions (ça prend 0.1 seconde !)
                pred_crise = modele_classif.predict(df_patient)[0] 
                
                # Gestion de predict_proba selon le type de modèle (XGBoost vs autres)
                if hasattr(modele_classif, "predict_proba"):
                    prob_crise = modele_classif.predict_proba(df_patient)[0][1] * 100
                else:
                    prob_crise = 100 if pred_crise == 1 else 0
                    
                pred_severite = modele_reg.predict(df_patient)[0]
                
                st.divider()
                st.subheader(" Résultats de l'Intelligence Artificielle")
                
                # 3. Affichage visuel des résultats
                col_res1, col_res2 = st.columns(2)
                
                with col_res1:
                    if pred_crise == 1:
                        st.error(f" **Haut Risque d'Exacerbation** (Probabilité : {prob_crise:.1f}%)")
                        st.write("Le patient nécessite une intervention clinique préventive.")
                    else:
                        st.success(f" **Risque Faible** (Probabilité : {prob_crise:.1f}%)")
                        st.write("Le patient est stable. Poursuivre le traitement actuel.")
                        
                with col_res2:
                    st.metric(
                        label="Nombre de crises estimées (365 jours)", 
                        value=f"{max(0, pred_severite):.1f} crises"
                    )

            except FileNotFoundError:
                st.error(" Les fichiers .pkl sont introuvables. Vérifiez qu'ils sont bien dans le dossier 'models/'.")
            except Exception as e:
                st.error(f" Une erreur est survenue lors de la prédiction : {e}")

    # --- ONGLET 3 : DONNÉES SYNTHÉTIQUES (MÉTHODE AVATAR) ---
    with tab_synth:
        st.write("**Générateur d'Avatars Cliniques (Privacy-by-Design)**")
        st.write("Génération basée sur l'algorithme des $k$-plus proches voisins (k-NN) avec création d'un barycentre aléatoire via la distribution de Dirichlet. Ce processus garantit une anonymisation totale tout en préservant la distribution statistique d'origine.")
        
        k_voisins = st.slider("Paramètre K (Modulation du compromis utilité/confidentialité)", 2, 10, 5)
        
        if st.button(f" Générer 5 Avatars avec K={k_voisins}"):
            
            # 1. Création d'une base source fictive
            colonnes = ["index_age", "female", "total_pre_index_cannisters_365", "pre_asthma_days", "total_pre_index_charge"]
            cols_binaires = ["female"]
            cols_entieres = ["pre_asthma_days", "total_pre_index_cannisters_365", "index_age"]
            
            df_source = pd.DataFrame({
                "index_age": np.random.normal(45, 15, 50),
                "female": np.random.choice([0, 1], 50),
                "total_pre_index_cannisters_365": np.random.normal(5, 4, 50),
                "pre_asthma_days": np.random.normal(20, 10, 50),
                "total_pre_index_charge": np.random.normal(3000, 1500, 50)
            })
            
            # 2. Standardisation
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_source)
            
            # 3. Ajustement k-NN
            knn = NearestNeighbors(n_neighbors=k_voisins)
            knn.fit(X_scaled)
            
            # On sélectionne 5 patients au hasard pour créer 5 avatars
            indices_sources = np.random.choice(range(len(X_scaled)), 5, replace=False)
            X_synth_scaled = np.zeros((5, X_scaled.shape[1]))
            
            # 4. Boucle de création
            for idx, i in enumerate(indices_sources):
                distances, indices = knn.kneighbors([X_scaled[i]])
                voisins = X_scaled[indices[0]] 
                
                # Barycentre aléatoire de Dirichlet
                poids = np.random.dirichlet(np.ones(k_voisins))
                X_synth_scaled[idx] = np.dot(poids, voisins)
                
            # 5. Retraitement
            X_synth = scaler.inverse_transform(X_synth_scaled)
            df_synth = pd.DataFrame(X_synth, columns=colonnes)
            
            # Restauration de la cohérence métier
            for col in colonnes:
                min_reel = df_source[col].min()
                df_synth[col] = np.maximum(df_synth[col], min_reel)
                if col in cols_binaires:
                    df_synth[col] = np.round(df_synth[col]).clip(0, 1)
                    
            for col in cols_entieres:
                if col in df_synth.columns:
                    df_synth[col] = np.round(df_synth[col])
                    
            df_synth.insert(0, 'patid', range(1000000, 1000000 + len(df_synth)))
            
            # 6. Affichage du résultat final (Plus de modèles IA ici !)
            st.success(" Avatars générés avec succès !")
            st.dataframe(df_synth, use_container_width=True)
            
            # Petit bonus : bouton pour télécharger le tableau généré
            csv = df_synth.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=" Télécharger ces avatars (CSV)",
                data=csv,
                file_name='patients_synthetiques.csv',
                mime='text/csv',
            )