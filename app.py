import base64
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Le modèle de classification est un objet utils.AsthmaXGB (sauvegardé avec scikit-learn 1.9.1)
import utils  # noqa: F401

ROOT = Path(__file__).parent
PORTFOLIO_URL = "https://abelard01.github.io"
GITHUB_URL = "https://github.com/Abelard01"
LINKEDIN_URL = "https://www.linkedin.com/in/abelard-mugisha"
EMAIL = "abelard.mugisha@etu.unistra.fr"

st.set_page_config(
    page_title="Démos IA santé · Abelard Mugisha",
    page_icon=":material/neurology:",
    layout="wide",
)

# --------------------------------------------------------------------------
# Résultats mesurés (sources : notebooks et dépôts du projet, recalculés)
# --------------------------------------------------------------------------
# DiagnosHand : matrices de confusion sur le jeu de test (113 patients)
DH = {
    "CNN EfficientNetV2-S": {"tn": 21, "fp": 41, "fn": 1, "tp": 50, "auc": 0.80},
    "BiomedCLIP": {"tn": 36, "fp": 26, "fn": 2, "tp": 49, "auc": 0.91},
}
# Asthme : jeu de test de 2 915 patients (20 %, stratifié, random_state=42)
ASTHME_CLF = {"n": 2915, "prev": 0.1139, "auc": 0.543, "ap": 0.126, "recall": 0.858,
              "precision": 0.127, "spec": 0.239, "tn": 618, "fp": 1965, "fn": 47, "tp": 285}
ASTHME_REG = {"dev": 0.9323, "dev_base": 0.9503, "rmse": 0.73, "mae": 0.33}
MOYENNE_CRISES = 0.175  # moyenne observée dans la cohorte de 14 575 patients

COMORBIDITES = {
    "pneumonia": "Pneumonie",
    "sinusitis": "Sinusite",
    "acute_bronchitis": "Bronchite aiguë",
    "acute_laryngitis": "Laryngite aiguë",
    "upper_respiratory_infection": "Infection des voies respiratoires sup.",
    "gerd": "Reflux gastro-œsophagien (RGO)",
    "rhinitis": "Rhinite",
}
LIBELLES = {
    **COMORBIDITES,
    "index_age": "Âge",
    "female": "Sexe",
    "total_pre_index_cannisters_365": "Bonbonnes de SABA",
    "pre_asthma_days": "Jours de traitement de l'asthme",
    "adherence": "Adhérence au traitement",
    "drug_s": "Indicateur drug_s",
    "previous_asthma_drugs": "Traitements antérieurs",
    "total_pre_index_charge": "Frais totaux",
    "pre_asthma_charge": "Frais asthme (hors pharmacie)",
    "pre_asthma_pharma_charge": "Frais pharmaceutiques asthme",
    "charge_par_jour": "Frais asthme par jour de traitement",
    "ratio_charge_asthme": "Part des frais liés à l'asthme",
}

# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&display=swap');
.block-container{padding-top:3.2rem;max-width:1180px}
h1,h2,h3{letter-spacing:-.02em}
.serif{font-family:'DM Serif Display',Georgia,serif;font-style:italic;font-weight:400;color:#1F4FD1}
.eyebrow{display:inline-block;font-family:'IBM Plex Mono',monospace;font-size:.78rem;font-weight:500;
  letter-spacing:.06em;color:#1F4FD1;background:#fff;border:1px solid #D8E0EC;border-radius:999px;padding:4px 12px;margin-bottom:.4rem}
.hero{display:flex;gap:28px;align-items:center;margin:.2rem 0 1.4rem}
.hero img{width:132px;height:132px;object-fit:cover;border-radius:24px;border:1px solid #D8E0EC;flex:none}
.hero h1{font-size:2.6rem;line-height:1.08;margin:0 0 .3rem;padding:0}
.hero p{color:#4A5568;font-size:1.08rem;margin:.2rem 0 0;max-width:46rem}
@media(max-width:700px){.hero{flex-direction:column;align-items:flex-start}.hero h1{font-size:2rem}}
.card{background:#fff;border:1px solid #D8E0EC;border-radius:18px;padding:20px 22px;height:100%}
.card h3{margin:.2rem 0 .4rem;font-size:1.3rem}
.card p{color:#4A5568;margin:0 0 .6rem}
.tag{display:inline-block;font-family:'IBM Plex Mono',monospace;font-size:.72rem;color:#4A5568;
  border:1px solid #D8E0EC;border-radius:999px;padding:2px 9px;margin:0 4px 4px 0;background:#F9FBFE}
.decision{border-left:3px solid #1F4FD1;padding:2px 0 2px 14px;margin:.4rem 0 1rem}
.decision .lbl{font-family:'IBM Plex Mono',monospace;font-size:.72rem;font-weight:500;letter-spacing:.1em;color:#1F4FD1;display:block}
.verdict{border-radius:16px;padding:16px 18px;border:1px solid}
.verdict.haut{background:#FCEBEB;border-color:#F09595;color:#791F1F}
.verdict.bas{background:#EAF3DE;border-color:#97C459;color:#27500A}
.verdict b{font-size:1.15rem}
.small{font-size:.85rem;color:#6B778A}
div[data-testid="stMetric"]{background:#fff;border:1px solid #D8E0EC;border-radius:14px;padding:12px 16px}
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Utilitaires
# --------------------------------------------------------------------------
@st.cache_data
def image_base64(chemin: str) -> str:
    return base64.b64encode((ROOT / chemin).read_bytes()).decode()


@st.cache_resource
def charger_modeles():
    clf = joblib.load(ROOT / "models" / "modele_classification.pkl")
    reg = joblib.load(ROOT / "models" / "modele_regression.pkl")
    return clf, reg


def eyebrow(texte: str):
    st.markdown(f'<span class="eyebrow">{texte}</span>', unsafe_allow_html=True)


def titre(texte: str, accent: str = ""):
    st.markdown(f"## {texte} <span class='serif'>{accent}</span>", unsafe_allow_html=True)


def pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f} %".replace(".", ",")


def sens_spec(m: dict) -> tuple[float, float]:
    return m["tp"] / (m["tp"] + m["fn"]), m["tn"] / (m["tn"] + m["fp"])


def avertissement():
    st.caption(
        ":material/info: Démonstration pédagogique. Ces outils ne sont pas des dispositifs médicaux "
        "et ne remplacent pas l'avis d'un professionnel de santé."
    )


# --------------------------------------------------------------------------
# Page : accueil
# --------------------------------------------------------------------------
def page_accueil():
    photo = image_base64("images/maphoto.JPG")
    st.markdown(
        f"""
<div class="hero">
  <img src="data:image/jpeg;base64,{photo}" alt="Photo d'Abelard Mugisha">
  <div>
    <span class="eyebrow">DÉMOS INTERACTIVES</span>
    <h1>Abelard Mugisha <span class="serif">· data scientist santé</span></h1>
    <p>Ingénieur biomédical et étudiant en Master Intelligence des données de santé (Université de Strasbourg).
    Ici, vous pouvez tester mes modèles et voir leurs performances réelles, y compris leurs limites.</p>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.link_button("Portfolio complet", PORTFOLIO_URL, icon=":material/language:", type="primary", width="stretch")
    c2.link_button("LinkedIn", LINKEDIN_URL, icon=":material/person:", width="stretch")
    c3.link_button("GitHub", GITHUB_URL, icon=":material/code:", width="stretch")
    c4.link_button("M'écrire", f"mailto:{EMAIL}", icon=":material/mail:", width="stretch")

    st.write("")
    eyebrow("PROJETS")
    titre("Deux projets,", "deux façons de valider un modèle.")

    clip = DH["BiomedCLIP"]
    sens, _ = sens_spec(clip)
    a, b = st.columns(2, gap="medium")
    with a:
        st.markdown(
            f"""
<div class="card">
  <span class="tag">VISION</span><span class="tag">MLOPS</span><span class="tag">HACKING HEALTH 2026</span>
  <h3>DiagnosHand : triage des plaies de la main</h3>
  <p>À partir d'une photo, orienter le patient vers une suture simple ou une exploration chirurgicale.</p>
  <p><b>AUC 0,91</b> · <b>sensibilité {pct(sens)}</b> · <b>15 chirurgies inutiles évitées</b> sur 113 patients.</p>
</div>""",
            unsafe_allow_html=True,
        )
        st.page_link(PAGES["diagnoshand"], label="Voir le projet", icon=":material/arrow_forward:")
    with b:
        st.markdown(
            f"""
<div class="card">
  <span class="tag">DONNÉES DE VIE RÉELLE</span><span class="tag">XGBOOST</span><span class="tag">RGPD</span>
  <h3>Asthme : risque d'exacerbation et données synthétiques</h3>
  <p>Prédire les crises de l'année suivante chez 14 575 patients, et générer des avatars qui protègent la vie privée.</p>
  <p>Simulateur branché sur les <b>vrais modèles</b>, avec une évaluation honnête (AUC {ASTHME_CLF['auc']:.2f}).</p>
</div>""".replace("0.54", "0,54"),
            unsafe_allow_html=True,
        )
        st.page_link(PAGES["asthme"], label="Tester le simulateur", icon=":material/arrow_forward:")

    st.write("")
    avertissement()


# --------------------------------------------------------------------------
# Page : DiagnosHand
# --------------------------------------------------------------------------
def page_diagnoshand():
    eyebrow("01 · VISION MÉDICALE · STAGE SOS MAIN")
    titre("DiagnosHand :", "triage IA des plaies de la main.")
    st.markdown(
        "Aux urgences de la main, il faut décider vite : **une suture simple** par l'urgentiste, "
        "ou **une exploration chirurgicale** au bloc. À partir d'une photo de la plaie, le modèle "
        "aide à orienter le patient. Projet né au Hacking Health Camp 2026 (lauréat), poursuivi en stage chez SOS Main."
    )
    st.markdown(
        """<div class="decision"><span class="lbl">DÉCISION CLÉ</span>
Rater une lésion grave coûte bien plus cher qu'une exploration de trop. Le seuil de décision est donc fixé à
<b>0,35</b> (indice de Youden) pour privilégier la sensibilité. Ensuite, le passage d'un CNN à un modèle
multimodal image + texte (BiomedCLIP) a réduit les faux positifs.</div>""",
        unsafe_allow_html=True,
    )

    onglet_res, onglet_demo, onglet_archi = st.tabs(
        [":material/monitoring: Résultats validés", ":material/photo_camera: Démo de l'interface",
         ":material/account_tree: Architecture et MLOps"]
    )

    with onglet_res:
        clip, cnn = DH["BiomedCLIP"], DH["CNN EfficientNetV2-S"]
        s_clip, sp_clip = sens_spec(clip)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("AUC", "0,91", "+0,11 vs CNN", help="Aire sous la courbe ROC, mesurée en validation.")
        m2.metric("Sensibilité", pct(s_clip), help="Jeu de test : 49 explorations nécessaires détectées sur 51.")
        m3.metric("Spécificité", pct(sp_clip), f"+{(sp_clip - sens_spec(cnn)[1]) * 100:.1f} pts vs CNN".replace(".", ","),
                  help="Jeu de test : 36 sutures simples correctement orientées sur 62.")
        m4.metric("Blocs évités", f"{cnn['fp'] - clip['fp']} / 113",
                  help="Explorations chirurgicales inutiles évitées : faux positifs du CNN (41) moins ceux de BiomedCLIP (26), sur 113 patients de test.")

        st.write("")
        st.markdown("##### Comparaison sur le jeu de test (113 patients)")
        tableau = {}
        for nom, m in DH.items():
            s, sp = sens_spec(m)
            tableau["CNN (référence)" if nom.startswith("CNN") else "BiomedCLIP (retenu)"] = {
                "Explorations détectées": f"{m['tp']} / {m['tp'] + m['fn']}",
                "Sensibilité": pct(s),
                "Spécificité": pct(sp),
                "Faux négatifs (% des patients)": pct(m["fn"] / 113),
                "Faux positifs (% des patients)": pct(m["fp"] / 113),
            }
        st.dataframe(pd.DataFrame(tableau), width="stretch")
        st.markdown(
            "<p class='small'>Le CNN rate 1 patient grave au lieu de 2, mais envoie 41 patients au bloc pour rien au lieu de 26. "
            "BiomedCLIP offre le meilleur compromis entre sécurité du patient et charge du bloc opératoire.</p>",
            unsafe_allow_html=True,
        )
        g, d = st.columns(2)
        g.image(str(ROOT / "images" / "confusion_cnn.png"), caption="CNN EfficientNetV2-S (référence)")
        d.image(str(ROOT / "images" / "confusion_biomedclip.png"), caption="BiomedCLIP (modèle retenu)")
        st.caption("Données : 561 photos exploitables (Pixacare), annotées par l'équipe ; découpage stratifié 448 / 113. "
                   "AUC et F1 (0,82) mesurés en validation ; matrices de confusion sur le jeu de test.")

    with onglet_demo:
        st.info(
            "Les poids du modèle et les photos de patients ne peuvent pas être publiés (données de santé). "
            "Cette page montre donc **l'interface et le traitement de l'image**, sans fausse prédiction : "
            "aucun résultat n'est inventé.",
            icon=":material/lock:",
        )
        fichier = st.file_uploader("Déposez une photo de main (JPG ou PNG)", type=["jpg", "jpeg", "png"])
        if fichier is not None:
            image = Image.open(fichier).convert("RGB")
            g, d = st.columns(2, gap="medium")
            with g:
                st.image(image, caption=f"Photo reçue · {image.width} × {image.height} px", width="stretch")
            with d:
                st.markdown("##### Ce que fait l'API à la réception")
                st.markdown(
                    "1. Vérifie le format (JPEG, PNG ou WebP) et décode l'image\n"
                    "2. Redimensionne au format attendu par le modèle\n"
                    "3. Normalise les couleurs (moyennes ImageNet)\n"
                    "4. Calcule la probabilité d'exploration et la compare au seuil **0,35**"
                )
                apercu = image.resize((224, 224))
                st.image(apercu, caption="Entrée de BiomedCLIP : 224 × 224 px", width=224)
            st.markdown("##### Format de la réponse de l'API (`POST /predict`)")
            st.code(
                '{\n'
                f'  "filename": "{fichier.name}",\n'
                '  "prediction": 0 | 1,              // 0 = suture simple, 1 = exploration\n'
                '  "label": "non" | "oui",\n'
                '  "confidence": 0.0–1.0,\n'
                '  "probability_exploration": 0.0–1.0,\n'
                '  "threshold_used": 0.35\n'
                '}',
                language="text",
            )

    with onglet_archi:
        st.markdown(
            """
| Étape | Choix technique |
|---|---|
| Données | 561 photos annotées, découpage stratifié 448 / 113 |
| Modèle de référence | CNN EfficientNetV2-S (PyTorch, transfer learning) |
| Modèle retenu | BiomedCLIP (image + texte clinique) + tête MLP, affiné en 2 phases |
| Décision | Seuil 0,35 (indice de Youden), orienté sensibilité |
| API | FastAPI : routes `/health` et `/predict` |
| Qualité | Tests Pytest, lint Ruff, CI GitHub Actions, gestion des dépendances avec uv |
| Déploiement | Dockerfile ; inférence en moins d'une seconde ; tests en conditions réelles à Strasbourg |
"""
        )
        st.caption("L'API publiée sur GitHub sert aujourd'hui le CNN de référence ; l'intégration de BiomedCLIP est l'étape suivante.")
        st.link_button("Voir le code de l'API", "https://github.com/Abelard01/HHC2026-DiagnosHand", icon=":material/code:")

    st.divider()
    avertissement()


# --------------------------------------------------------------------------
# Page : Asthme
# --------------------------------------------------------------------------
# Profils d'exemple (fictifs) : valeurs médianes de la cohorte, puis un score bas et un score élevé
PROFILS = {
    "Patient médian": dict(age=41, femme=True, saba=1, jours=1, adh=0.17, drug_s=False, comorb=[],
                           total=3170, asthme=131, pharma=46),
    "Exemple sous le seuil": dict(age=62, femme=False, saba=2, jours=2, adh=0.08, drug_s=False, comorb=[],
                                  total=1200, asthme=131, pharma=600),
    "Exemple au-dessus du seuil": dict(age=55, femme=True, saba=2, jours=8, adh=0.75, drug_s=True,
                                       comorb=["pneumonia", "sinusitis", "gerd"], total=25000, asthme=2500, pharma=1100),
}


def appliquer_profil(nom: str):
    p = PROFILS[nom]
    for cle, valeur in p.items():
        st.session_state[f"a_{cle}"] = valeur


def page_asthme():
    eyebrow("02 · DONNÉES DE VIE RÉELLE · DATA CHALLENGE M1")
    titre("Asthme :", "anticiper les crises de l'année suivante.")
    st.markdown(
        "Cohorte de **14 575 patients asthmatiques** (données de facturation et de soins de l'année précédente). "
        "Deux modèles XGBoost : une **classification** (au moins une crise dans l'année ?) et une "
        "**régression de Poisson** (combien de crises ?)."
    )

    if "a_age" not in st.session_state:
        appliquer_profil("Patient médian")

    o_sim, o_perf, o_synth = st.tabs(
        [":material/person: Simulateur patient", ":material/fact_check: Performance réelle",
         ":material/shield_person: Données synthétiques"]
    )

    with o_sim:
        st.markdown("**Partir d'un profil type**")
        cols = st.columns(len(PROFILS))
        for col, nom in zip(cols, PROFILS):
            col.button(nom, on_click=appliquer_profil, args=(nom,), width="stretch")

        with st.form("patient"):
            st.markdown("**Patient**")
            c1, c2, c3 = st.columns(3)
            age = c1.number_input("Âge (ans)", 12, 65, key="a_age", help="Cohorte de 12 à 65 ans.")
            femme = c2.toggle("Femme", key="a_femme")
            drug_s = c3.toggle("Indicateur drug_s", key="a_drug_s",
                               help="Variable binaire du jeu de données, non décrite dans le dictionnaire du challenge "
                                    "(probablement le groupe de traitement attribué).")

            st.markdown("**Année précédente**")
            c1, c2, c3 = st.columns(3)
            saba = c1.number_input("Bonbonnes de SABA", 0, 2, key="a_saba",
                                   help="Bronchodilatateurs de secours. Distribution tronquée à 2 par les critères d'inclusion.")
            jours = c2.number_input("Jours avec traitement de l'asthme", 0, 100, key="a_jours")
            adh = c3.slider("Adhérence au traitement", 0.0, 1.0, step=0.01, key="a_adh",
                            help="Score de 0 à 1. Médiane de la cohorte : 0,17.")
            comorb = st.multiselect("Diagnostics de l'année précédente", list(COMORBIDITES),
                                    format_func=COMORBIDITES.get, key="a_comorb",
                                    placeholder="Aucun diagnostic associé")

            st.markdown("**Frais de santé (en dollars)**")
            c1, c2, c3 = st.columns(3)
            total = c1.number_input("Frais totaux (12 mois)", 0, 900_000, step=100, key="a_total")
            asthme = c2.number_input("Frais asthme hors pharmacie (6 mois)", 0, 80_000, step=10, key="a_asthme")
            pharma = c3.number_input("Frais pharmaceutiques asthme (12 mois)", 0, 6_000, step=10, key="a_pharma")

            lancer = st.form_submit_button("Calculer le risque", type="primary", icon=":material/play_arrow:")

        if lancer:
            patient = {
                "index_age": age, "female": int(femme),
                **{k: int(k in comorb) for k in COMORBIDITES},
                "previous_asthma_drugs": 1,  # constante dans toute la cohorte
                "total_pre_index_cannisters_365": saba, "adherence": adh, "pre_asthma_days": jours,
                "drug_s": int(drug_s), "total_pre_index_charge": total, "pre_asthma_charge": asthme,
                "pre_asthma_pharma_charge": pharma,
                "log_charges": np.log(total) if total > 0 else np.nan,
                "log_asthma_charge": np.log(asthme) if asthme > 0 else np.nan,
            }
            df = pd.DataFrame([patient])
            try:
                clf, reg = charger_modeles()
                proba = float(clf.predict_proba(df)[0][1])
                seuil = float(clf.best_threshold)
                crises = max(0.0, float(reg.predict(df)[0]))
            except Exception as e:  # le modèle ne doit jamais faire planter la page
                st.error(f"Le modèle n'a pas pu calculer ce profil : {e}")
                return

            st.write("")
            g, d = st.columns([3, 2], gap="medium")
            with g:
                alerte = proba >= seuil
                st.markdown(
                    f"""<div class="verdict {'haut' if alerte else 'bas'}">
<b>{'Alerte : patient à surveiller' if alerte else 'Pas d’alerte'}</b><br>
Score du modèle : <b>{pct(proba)}</b> · seuil d'alerte : {pct(seuil)}</div>""",
                    unsafe_allow_html=True,
                )
                donnees = pd.DataFrame({"x": [0.0], "x2": [proba]})
                barre = alt.Chart(donnees).mark_bar(height=22, cornerRadius=6,
                                                     color="#E24B4A" if alerte else "#639922").encode(
                    x=alt.X("x:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", title=None)), x2="x2:Q")
                fond = alt.Chart(pd.DataFrame({"a": [0.0], "b": [1.0]})).mark_bar(height=22, cornerRadius=6,
                                                                                   color="#E6EEFF").encode(x="a:Q", x2="b:Q")
                regle = alt.Chart(pd.DataFrame({"s": [seuil], "t": ["seuil"]})).mark_rule(color="#0F1B2D", strokeWidth=2).encode(x="s:Q")
                st.altair_chart((fond + barre + regle).properties(height=60), width="stretch")
                st.markdown(
                    f"<p class='small'>Le seuil ({pct(seuil)}) a été choisi pour maximiser le F1 : il privilégie la détection "
                    f"(86 % des patients qui feront une crise sont alertés) au prix de nombreuses fausses alertes. "
                    f"Ce score n'est pas une probabilité calibrée.</p>",
                    unsafe_allow_html=True,
                )
            with d:
                st.metric("Crises attendues sur 12 mois", f"{crises:.2f}".replace(".", ","),
                          f"{crises - MOYENNE_CRISES:+.2f} vs moyenne".replace(".", ","),
                          delta_color="inverse")
                st.caption(f"Moyenne observée dans la cohorte : {MOYENNE_CRISES:.3f} crise par patient.".replace(".", ",", 1))

    with o_perf:
        p = ASTHME_CLF
        st.markdown(
            """<div class="decision"><span class="lbl">CE QUE LES CHIFFRES DISENT</span>
Ces modèles détectent la plupart des patients à risque, mais <b>distinguent mal</b> les patients entre eux :
l'AUC est proche du hasard. Je les présente tels quels : savoir lire les limites d'un modèle fait partie du travail.</div>""",
            unsafe_allow_html=True,
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("AUC ROC", f"{p['auc']:.2f}".replace(".", ","), help="0,5 = hasard, 1 = parfait.")
        m2.metric("Rappel", pct(p["recall"], 0), help=f"{p['tp']} patients alertés sur {p['tp'] + p['fn']} qui ont fait une crise.")
        m3.metric("Précision", pct(p["precision"], 0), help="Part des alertes qui sont de vraies crises.")
        m4.metric("Spécificité", pct(p["spec"], 0))

        g, d = st.columns(2, gap="medium")
        with g:
            st.markdown(f"##### Matrice de confusion · {p['n']:,} patients de test".replace(",", " "))
            mc = pd.DataFrame({
                "Réalité": ["Pas de crise", "Pas de crise", "Crise", "Crise"],
                "Prédiction": ["Pas d'alerte", "Alerte", "Pas d'alerte", "Alerte"],
                "n": [p["tn"], p["fp"], p["fn"], p["tp"]],
            })
            base = alt.Chart(mc).encode(x=alt.X("Prédiction:N", sort=["Pas d'alerte", "Alerte"], title=None),
                                        y=alt.Y("Réalité:N", sort=["Pas de crise", "Crise"], title=None))
            carte = base.mark_rect(cornerRadius=8).encode(
                color=alt.Color("n:Q", scale=alt.Scale(range=["#E6EEFF", "#1F4FD1"]), legend=None))
            texte = base.mark_text(fontSize=18, fontWeight=600).encode(
                text="n:Q", color=alt.condition("datum.n > 1000", alt.value("white"), alt.value("#0F1B2D")))
            st.altair_chart((carte + texte).properties(height=230), width="stretch")
        with g:
            st.caption(f"Prévalence : {pct(p['prev'])} des patients font au moins une crise. "
                       "Découpage 80/20 stratifié (random_state = 42), seuil 0,4216.")
        with d:
            st.markdown("##### Variables les plus utilisées par le modèle")
            st.altair_chart(graphique_importance(), width="stretch")
            st.caption("Importance XGBoost (gain), regroupée par variable d'origine.")

        r = ASTHME_REG
        st.markdown("##### Régression du nombre de crises (Poisson)")
        r1, r2, r3 = st.columns(3)
        r1.metric("Déviance de Poisson", f"{r['dev']:.3f}".replace(".", ","),
                  f"{(r['dev'] - r['dev_base']) / r['dev_base'] * 100:.1f} % vs prédire la moyenne".replace(".", ","),
                  delta_color="inverse")
        r2.metric("Erreur absolue moyenne", f"{r['mae']:.2f} crise".replace(".", ","))
        r3.metric("RMSE", f"{r['rmse']:.2f}".replace(".", ","))

        st.markdown("##### Ce que j'en retiens")
        st.markdown(
            "- Les données de facturation seules prédisent mal les crises : il manque les **antécédents d'exacerbation**, "
            "la **spirométrie** ou les **éosinophiles**, qui sont les vrais prédicteurs cliniques.\n"
            "- Avec 11 % de patients positifs, l'accuracy est trompeuse : j'ai évalué le rappel, la précision et l'AUC.\n"
            "- Étapes suivantes : calibrer les probabilités, ajouter des variables cliniques, valider sur une cohorte externe."
        )

    with o_synth:
        st.markdown(
            "La **méthode Avatar** (Guillaudeux et al., 2023) crée pour chaque patient un « avatar » à partir de ses "
            "*k* plus proches voisins. Les avatars gardent les tendances statistiques de la cohorte sans correspondre "
            "à une vraie personne."
        )
        m1, m2 = st.columns(2)
        m1.metric("Distance de Wasserstein moyenne", "0,098", help="Écart entre les distributions réelles et synthétiques (0 = identiques).")
        m2.metric("Écart moyen des corrélations", "0,0105", help="Les liens entre variables sont conservés.")
        st.caption("Résultats mesurés sur la vraie cohorte pendant le projet.")

        st.markdown("##### Démonstration sur une cohorte fictive")
        st.caption("Pour ne publier aucune donnée réelle, la démo ci-dessous utilise 200 patients générés aléatoirement.")
        c1, c2 = st.columns([1, 3])
        k = c1.slider("Nombre de voisins k", 2, 20, 8, help="Plus k est grand, plus les avatars s'éloignent des vrais patients.")
        graine = c1.number_input("Graine aléatoire", 0, 999, 42)
        reel, avatars = generer_avatars(k, int(graine))
        nuage = pd.concat([reel.assign(Source="Patients fictifs"), avatars.assign(Source="Avatars")])
        graphe = alt.Chart(nuage).mark_circle(size=46, opacity=.7).encode(
            x=alt.X("Âge:Q", scale=alt.Scale(zero=False)),
            y=alt.Y("Frais totaux ($):Q", scale=alt.Scale(type="log")),
            color=alt.Color("Source:N", scale=alt.Scale(range=["#B4B2A9", "#1F4FD1"]), legend=alt.Legend(orient="top", title=None)),
            tooltip=["Source", "Âge", "Bonbonnes de SABA", "Frais totaux ($)"],
        ).properties(height=340)
        c2.altair_chart(graphe, width="stretch")
        st.dataframe(avatars.head(8).round({"Frais totaux ($)": 0}), hide_index=True, width="stretch")

    st.divider()
    avertissement()


@st.cache_data
def graphique_importance():
    clf, _ = charger_modeles()
    pre = clf.model.named_steps["preprocessor"]
    noms = pre.get_feature_names_out()
    imp = clf.model.named_steps["classifier"].feature_importances_
    lignes = []
    for nom, v in zip(noms, imp):
        brut = nom.split("__", 1)[1]
        origine = next((f for f in clf.cat_features if brut.startswith(f + "_")), brut)
        lignes.append({"Variable": LIBELLES.get(origine, origine), "Importance": float(v)})
    df = pd.DataFrame(lignes).groupby("Variable", as_index=False)["Importance"].sum()
    df = df.sort_values("Importance", ascending=False).head(10)
    return alt.Chart(df).mark_bar(color="#1F4FD1", cornerRadiusEnd=4).encode(
        x=alt.X("Importance:Q", axis=alt.Axis(format="%", title=None)),
        y=alt.Y("Variable:N", sort="-x", title=None, axis=alt.Axis(labelLimit=280)),
        tooltip=[alt.Tooltip("Importance:Q", format=".1%")],
    ).properties(height=300)


@st.cache_data
def generer_avatars(k: int, graine: int):
    rng = np.random.default_rng(graine)
    n = 200
    reel = pd.DataFrame({
        "Âge": rng.integers(12, 66, n),
        "Femme": rng.binomial(1, 0.63, n),
        "Bonbonnes de SABA": rng.choice([0, 1, 2], n, p=[0.36, 0.48, 0.16]),
        "Jours de traitement": rng.poisson(1.4, n),
        "Frais totaux ($)": np.round(np.exp(rng.normal(8.1, 1.2, n)), 0),
    })
    X = reel.assign(**{"Frais totaux ($)": np.log(reel["Frais totaux ($)"])})
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    voisins = NearestNeighbors(n_neighbors=k).fit(Xs).kneighbors(Xs, return_distance=False)
    poids = rng.dirichlet(np.ones(k), n)
    synth = scaler.inverse_transform(np.einsum("nk,nkd->nd", poids, Xs[voisins]))
    avatars = pd.DataFrame(synth, columns=X.columns)
    avatars["Frais totaux ($)"] = np.exp(avatars["Frais totaux ($)"]).round(0)
    for col in ["Âge", "Femme", "Bonbonnes de SABA", "Jours de traitement"]:
        avatars[col] = avatars[col].round().clip(lower=0).astype(int)
    return reel, avatars


# --------------------------------------------------------------------------
# Navigation
# --------------------------------------------------------------------------
PAGES = {
    "accueil": st.Page(page_accueil, title="Accueil", icon=":material/home:", default=True),
    "diagnoshand": st.Page(page_diagnoshand, title="DiagnosHand", icon=":material/back_hand:", url_path="diagnoshand"),
    "asthme": st.Page(page_asthme, title="Asthme", icon=":material/pulmonology:", url_path="asthme"),
}

with st.sidebar:
    st.markdown("**Abelard Mugisha**  \nData scientist santé · Strasbourg")
    st.link_button("Portfolio complet", PORTFOLIO_URL, icon=":material/language:", width="stretch")
    st.caption(f"[{EMAIL}](mailto:{EMAIL})  \n[LinkedIn]({LINKEDIN_URL}) · [GitHub]({GITHUB_URL})")

st.navigation(list(PAGES.values())).run()
