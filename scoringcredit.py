
# ============================================================
# CREDIT SCORING DASHBOARD
# XGBoost + MLP + Modèle Hybride
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import xgboost as xgb
import tensorflow as tf
from keras.models import load_model


# ============================================================
# 1. CONFIGURATION STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Credit Scoring Dashboard",
    page_icon="💳",
    layout="wide"
)


# ============================================================
# 2. CHARGEMENT DES MODÈLES ET DES PARAMÈTRES
# ============================================================

@st.cache_resource
def load_all_assets():

    try:

        # ----------------------------------------------------
        # Scaler
        # ----------------------------------------------------

        with open(
            "feature_scaler.pkl",
            "rb"
        ) as f:

            scaler = pickle.load(f)


        # ----------------------------------------------------
        # Liste des variables
        # ----------------------------------------------------

        with open(
            "feature_columns.json",
            "r",
            encoding="utf-8"
        ) as f:

            feature_columns = json.load(f)


        # ----------------------------------------------------
        # MLP
        # ----------------------------------------------------

        mlp_model = load_model(
            "mlp_credit_model.keras",
            compile=False
        )


        # ----------------------------------------------------
        # XGBoost
        # ----------------------------------------------------
        # Load XGBoost model (from JSON)
        # Charger le modèle XGBoost depuis un fichier JSON
        
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model("xgboost_credit_model.json")

        # ----------------------------------------------------
        # Modèle hybride
        # ----------------------------------------------------

        with open(
            "hybrid_model.pkl",
            "rb"
        ) as f:

            meta_model = pickle.load(f)


        return (
            scaler,
            feature_columns,
            mlp_model,
            xgb_model,
            meta_model
        )


    except Exception as e:

        st.error(
            "❌ Impossible de charger les modèles."
        )

        st.exception(e)

        st.stop()


(
    scaler,
    feature_columns,
    mlp_model,
    xgb_model,
    meta_model
) = load_all_assets()


# ============================================================
# 3. FONCTION DE SEGMENTATION DU RISQUE
# ============================================================

def assign_credit_zone(
    probability,
    green_threshold=0.20,
    red_threshold=0.70
):

    if probability < green_threshold:

        return (
            "VERT",
            "Acceptation automatique",
            "🟢"
        )

    elif probability < red_threshold:

        return (
            "ORANGE",
            "Analyse manuelle",
            "🟠"
        )

    else:

        return (
            "ROUGE",
            "Refus automatique ou contrôle renforcé",
            "🔴"
        )


# ============================================================
# 4. TITRE
# ============================================================

st.title(
    "💳 Credit Scoring Dashboard"
)

st.write(
    "Système intelligent d'évaluation du risque crédit "
    "reposant sur XGBoost, un réseau de neurones MLP "
    "et un modèle hybride."
)


# ============================================================
# 5. INFORMATIONS SUR LES MODÈLES
# ============================================================

with st.expander(
    "ℹ️ Modèles utilisés"
):

    col1, col2, col3 = st.columns(3)

    with col1:

        st.info(
            "**XGBoost**\n\n"
            "Modélisation des relations non linéaires "
            "dans les données tabulaires."
        )

    with col2:

        st.info(
            "**MLP**\n\n"
            "Réseau de neurones permettant "
            "d'apprendre des représentations non linéaires."
        )

    with col3:

        st.info(
            "**Modèle hybride**\n\n"
            "Combinaison des probabilités produites "
            "par XGBoost et le MLP."
        )


# ============================================================
# 6. SAISIE DES INFORMATIONS DU CLIENT
# ============================================================

st.sidebar.header(
    "👤 Informations du client"
)

st.sidebar.write(
    "Veuillez saisir les informations disponibles "
    "dans le dossier de crédit."
)


input_data = {}


for feature in feature_columns:

    label = feature.replace(
        "_",
        " "
    )


    # --------------------------------------------------------
    # Âge
    # --------------------------------------------------------

    if feature == "age":

        input_data[feature] = st.sidebar.number_input(
            label,
            min_value=18,
            max_value=100,
            value=35,
            step=1,
            key=f"input_{feature}"
        )


    # --------------------------------------------------------
    # Revenu mensuel
    # --------------------------------------------------------

    elif feature == "MonthlyIncome":

        input_data[feature] = st.sidebar.number_input(
            label,
            min_value=0.0,
            max_value=1_000_000.0,
            value=5_000.0,
            step=100.0,
            key=f"input_{feature}"
        )


    # --------------------------------------------------------
    # Nombre de personnes à charge
    # --------------------------------------------------------

    elif feature == "NumberOfDependents":

        input_data[feature] = st.sidebar.number_input(
            label,
            min_value=0,
            max_value=20,
            value=0,
            step=1,
            key=f"input_{feature}"
        )


    # --------------------------------------------------------
    # Ratio d'endettement
    # --------------------------------------------------------

    elif feature == "DebtRatio":

        input_data[feature] = st.sidebar.number_input(
            label,
            min_value=0.0,
            max_value=10_000.0,
            value=0.30,
            step=0.01,
            key=f"input_{feature}"
        )


    # --------------------------------------------------------
    # Utilisation du crédit
    # --------------------------------------------------------

    elif feature == (
        "RevolvingUtilizationOfUnsecuredLines"
    ):

        input_data[feature] = st.sidebar.number_input(
            label,
            min_value=0.0,
            max_value=100.0,
            value=0.50,
            step=0.01,
            key=f"input_{feature}"
        )


    # --------------------------------------------------------
    # Variables binaires
    # --------------------------------------------------------

    elif feature == "HasSeriousDelinquency":

        input_data[feature] = st.sidebar.selectbox(
            label,
            [0, 1],
            key=f"input_{feature}"
        )


    # --------------------------------------------------------
    # Autres variables numériques
    # --------------------------------------------------------

    else:

        input_data[feature] = st.sidebar.number_input(
            label,
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0,
            key=f"input_{feature}"
        )


# ============================================================
# 7. BOUTON DE PRÉDICTION
# ============================================================

predict_button = st.sidebar.button(
    "🔍 Évaluer le risque",
    type="primary",
    use_container_width=True
)


# ============================================================
# 8. PRÉDICTION
# ============================================================

if predict_button:

    try:

        # ====================================================
        # 8.1 Création du DataFrame
        # ====================================================

        input_df = pd.DataFrame(
            [input_data]
        )


        # ====================================================
        # 8.2 FEATURE ENGINEERING
        # ====================================================

        # ----------------------------------------------------
        # CreditHistoryLength
        # ----------------------------------------------------

        if "CreditHistoryLength" in feature_columns:

            input_df[
                "CreditHistoryLength"
            ] = (
                input_df["age"] - 18
            )


        # ----------------------------------------------------
        # TotalPastDue
        # ----------------------------------------------------

        past_due_columns = [
            "NumberOfTime30-59DaysPastDueNotWorse",
            "NumberOfTime60-89DaysPastDueNotWorse",
            "NumberOfTimes90DaysLate"
        ]

        existing_past_due = [
            col
            for col in past_due_columns
            if col in input_df.columns
        ]

        if (
            "TotalPastDue" in feature_columns
            and len(existing_past_due) > 0
        ):

            input_df["TotalPastDue"] = (
                input_df[
                    existing_past_due
                ]
                .sum(axis=1)
            )


        # ----------------------------------------------------
        # HasSeriousDelinquency
        # ----------------------------------------------------

        if (
            "HasSeriousDelinquency"
            in feature_columns
        ):

            if (
                "NumberOfTimes90DaysLate"
                in input_df.columns
            ):

                input_df[
                    "HasSeriousDelinquency"
                ] = (
                    input_df[
                        "NumberOfTimes90DaysLate"
                    ] > 0
                ).astype(int)


        # ====================================================
        # 8.3 VÉRIFICATION DES VARIABLES
        # ====================================================

        missing_features = [
            feature
            for feature in feature_columns
            if feature not in input_df.columns
        ]

        if missing_features:

            st.error(
                "❌ Variables manquantes : "
                + ", ".join(missing_features)
            )

            st.stop()


        # ====================================================
        # 8.4 ORDRE EXACT DES VARIABLES
        # ====================================================

        input_df_ordered = (
            input_df[
                feature_columns
            ].copy()
        )


        # ====================================================
        # 8.5 CONVERSION NUMÉRIQUE
        # ====================================================

        input_df_ordered = (
            input_df_ordered
            .apply(
                pd.to_numeric,
                errors="coerce"
            )
        )


        if input_df_ordered.isnull().any().any():

            st.error(
                "❌ Certaines valeurs sont invalides."
            )

            st.dataframe(
                input_df_ordered
            )

            st.stop()


        # ====================================================
        # 8.6 STANDARDISATION
        # ====================================================

        input_scaled = (
            scaler.transform(
                input_df_ordered
            )
        )


        # ====================================================
        # 8.7 PRÉDICTION XGBOOST
        # ====================================================

        xgb_proba = float(
            xgb_model
            .predict_proba(
                input_scaled
            )[0, 1]
        )


        # ====================================================
        # 8.8 PRÉDICTION MLP
        # ====================================================

        mlp_proba = float(
            mlp_model
            .predict(
                input_scaled,
                verbose=0
            )[0, 0]
        )


        # ====================================================
        # 8.9 MODÈLE HYBRIDE
        # ====================================================

        stacked_input = np.array(
            [
                [
                    mlp_proba,
                    xgb_proba
                ]
            ]
        )


        hybrid_proba = float(
            meta_model
            .predict_proba(
                stacked_input
            )[0, 1]
        )


        # ====================================================
        # 8.10 ZONE DE RISQUE
        # ====================================================

        (
            credit_zone,
            decision,
            zone_icon
        ) = assign_credit_zone(
            hybrid_proba
        )


        # ====================================================
        # 9. RÉSULTATS
        # ====================================================

        st.divider()

        st.header(
            "📊 Résultats de l'évaluation"
        )


        # ----------------------------------------------------
        # Comparaison des modèles
        # ----------------------------------------------------

        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "XGBoost",
                f"{xgb_proba:.2%}"
            )


        with col2:

            st.metric(
                "MLP",
                f"{mlp_proba:.2%}"
            )


        with col3:

            st.metric(
                "Modèle hybride",
                f"{hybrid_proba:.2%}"
            )


        # ====================================================
        # 10. DÉCISION
        # ====================================================

        st.divider()

        st.subheader(
            "🎯 Décision de risque"
        )


        if credit_zone == "VERT":

            st.success(
                f"🟢 ZONE VERTE\n\n"
                f"Probabilité de défaut : "
                f"**{hybrid_proba:.2%}**\n\n"
                f"Décision proposée : "
                f"**{decision}**"
            )


        elif credit_zone == "ORANGE":

            st.warning(
                f"🟠 ZONE ORANGE\n\n"
                f"Probabilité de défaut : "
                f"**{hybrid_proba:.2%}**\n\n"
                f"Décision proposée : "
                f"**{decision}**"
            )


        else:

            st.error(
                f"🔴 ZONE ROUGE\n\n"
                f"Probabilité de défaut : "
                f"**{hybrid_proba:.2%}**\n\n"
                f"Décision proposée : "
                f"**{decision}**"
            )


        # ====================================================
        # 11. INTERPRÉTATION
        # ====================================================

        st.divider()

        st.subheader(
            "📋 Interprétation de la décision"
        )


        if credit_zone == "VERT":

            st.write(
                "Le profil présente une faible probabilité "
                "estimée de défaut. Le dossier peut être "
                "orienté vers une procédure d'acceptation "
                "automatique, sous réserve des règles "
                "réglementaires et opérationnelles."
            )


        elif credit_zone == "ORANGE":

            st.write(
                "Le profil présente un niveau de risque "
                "intermédiaire. Une analyse complémentaire "
                "par un analyste crédit est recommandée "
                "avant toute décision."
            )


        else:

            st.write(
                "Le profil présente une probabilité élevée "
                "de défaut. Le dossier doit faire l'objet "
                "d'un refus automatique ou d'un contrôle "
                "renforcé selon la politique de crédit."
            )


        # ====================================================
        # 12. INFORMATIONS DU CLIENT
        # ====================================================

        with st.expander(
            "👤 Voir les données utilisées pour la prédiction"
        ):

            st.dataframe(
                input_df_ordered,
                use_container_width=True
            )


        # ====================================================
        # 13. RAPPEL DES SEUILS
        # ====================================================

        with st.expander(
            "📌 Grille de décision"
        ):

            threshold_df = pd.DataFrame({

                "Zone": [
                    "🟢 Verte",
                    "🟠 Orange",
                    "🔴 Rouge"
                ],

                "Probabilité": [
                    "< 20 %",
                    "20 % – 70 %",
                    "≥ 70 %"
                ],

                "Décision": [
                    "Acceptation automatique",
                    "Analyse manuelle",
                    "Refus / contrôle renforcé"
                ]

            })

            st.table(
                threshold_df
            )


    except Exception as e:

        st.error(
            "❌ Une erreur est survenue "
            "pendant la prédiction."
        )

        st.exception(e)


# ============================================================
# 14. PIED DE PAGE
# ============================================================

st.divider()

st.caption(
    "Système intelligent d'évaluation du risque crédit "
    "— XGBoost + MLP + modèle hybride"
)

