# Importation des outils de manipulation de données
import pandas as pd
import numpy as np

# Outils pour diviser nos données et créer nos pipelines de traitement
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.pipeline import Pipeline as ImbPipeline

# Les algorithmes et métriques que nous avons utilisés
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve
from xgboost import XGBClassifier

# outil de visualisation
import matplotlib.pyplot as plt
import seaborn as sns


class AsthmaXGB:
    def __init__(self):
        # Variables qualitatives
        self.cat_features = [
            'previous_asthma_drugs', 'pneumonia', 'sinusitis', 'acute_bronchitis',
            'acute_laryngitis', 'upper_respiratory_infection', 'gerd', 'rhinitis',
            'drug_s', 'female'
        ]
        # Variables quantitatives
        self.num_features = [
            'index_age', 'total_pre_index_cannisters_365', 'pre_asthma_days',
            'pre_asthma_charge', 'adherence', 'total_pre_index_charge',
            'pre_asthma_pharma_charge', 'charge_par_jour', 'ratio_charge_asthme'
        ]
        self.best_threshold = 0.5

        # Pipeline de prétraitement
        preprocessor = ColumnTransformer(transformers=[
            ('num', ImbPipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), self.num_features),
            ('cat', ImbPipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ]), self.cat_features)
        ])

        # Modèle XGBOOST avec poids pour équilibrer
        self.model = ImbPipeline([
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(
                scale_pos_weight=7.78,
                random_state=42,
                eval_metric='logloss'
            ))
        ])

    def feature_engineering(self, df):
        """ Version Pandas garantie sans erreur .select """
        df = df.copy()
        df["charge_par_jour"] = df["pre_asthma_charge"] / (df["pre_asthma_days"] + 1)
        df["ratio_charge_asthme"] = df["pre_asthma_charge"] / (df["total_pre_index_charge"] + 1)
        return df

    def find_best_threshold(self, y_test, y_proba):
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        self.best_threshold = thresholds[np.argmax(f1_scores)]
        print(f"Seuil optimisé : {self.best_threshold:.4f}")

    def train_and_evaluate(self, file_path):
        # Lecture en PANDAS
        df = pd.read_csv(file_path)
        df = self.feature_engineering(df)

        X = df[self.num_features + self.cat_features]
        y = (df['post_index_exacerbations365'] > 0).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [3, 5],
            'classifier__learning_rate': [0.01, 0.05]
        }

        grid_search = GridSearchCV(self.model, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_

        y_proba = self.model.predict_proba(X_test)[:, 1]
        self.find_best_threshold(y_test, y_proba)

        y_pred_custom = (y_proba >= self.best_threshold).astype(int)
        print("\nRÉSULTATS XGBOOST (ÉQUILIBRÉ) :")
        print(classification_report(y_test, y_pred_custom))

        return df

    def predict(self, df):
        """Prépare les données puis fait la prédiction (0 ou 1)"""
        # 1. On applique les calculs mathématiques (création des 2 colonnes manquantes)
        df_prepared = self.feature_engineering(df)
        
        # 2. On sélectionne uniquement les colonnes attendues par le modèle (dans le bon ordre)
        X_final = df_prepared[self.num_features + self.cat_features]
        
        # 3. On demande au modèle de prédire avec le seuil optimisé
        y_proba = self.model.predict_proba(X_final)[:, 1]
        return (y_proba >= self.best_threshold).astype(int)

    def predict_proba(self, df):
        """Prépare les données puis renvoie les probabilités (pourcentage)"""
        df_prepared = self.feature_engineering(df)
        X_final = df_prepared[self.num_features + self.cat_features]
        return self.model.predict_proba(X_final)

    def save_predictions(self, full_df, output_name="predictions_finales.csv"):
        X_final = full_df[self.num_features + self.cat_features]
        y_proba_final = self.model.predict_proba(X_final)[:, 1]
        preds = (y_proba_final >= self.best_threshold).astype(int)

        resultat = pd.DataFrame({
            "patid": full_df["patid"],
            "prediction": preds
        })
        resultat.to_csv(output_name, index=False)
        print(f"\n Fichier prêt : {output_name}")