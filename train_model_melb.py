"""
Script de entrenamiento de modelos de regresión para predecir Price
usando el dataset Melbourne Housing.

- Carga datos crudos desde data/melb_data.csv
- Aplica la misma imputación definida en utils_melb.impute_df
- Entrena dos modelos: Ridge y RandomForest
- Compara MAE y R² en un conjunto de prueba
- Guarda el MEJOR modelo como models/melb_model.pkl
- Guarda métricas en models/melb_metrics.json
- Guarda predicciones de prueba en models/melb_test_predictions.csv

Ejecutar desde la raíz del proyecto:

    python train_model_melb.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from utils_melb import load_raw, impute_df

# ============================
# 1. Rutas y carga de datos
# ============================

DATA_PATH = Path("data") / "melb_data.csv"
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

print(f"Cargando datos desde: {DATA_PATH}")
df_raw = load_raw(str(DATA_PATH))

print("Aplicando imputación definida en utils_melb.impute_df...")
df = impute_df(df_raw)

# ============================
# 2. Definir variables
# ============================

TARGET = "Price"

# Columnas que usaremos como predictores
NUMERIC_FEATURES = [
    "Rooms",
    "Distance",
    "Landsize",
    "BuildingArea",
    "Bedroom2",
    "Bathroom",
    "Car",
    "Propertycount",
    "YearBuilt",
    "Lattitude",
    "Longtitude",
]

CATEGORICAL_FEATURES = [
    "Type",
    "Method",
    "SellerG",
    "CouncilArea",
    "Regionname",
    "Suburb",
    "Postcode",
]

# Filtrar solo columnas disponibles (por si alguna no existe)
NUMERIC_FEATURES = [c for c in NUMERIC_FEATURES if c in df.columns]
CATEGORICAL_FEATURES = [c for c in CATEGORICAL_FEATURES if c in df.columns]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

print("Variables numéricas usadas:", NUMERIC_FEATURES)
print("Variables categóricas usadas:", CATEGORICAL_FEATURES)

# Eliminar filas sin Price o sin alguna feature importante
df_model = df.dropna(subset=[TARGET])
df_model = df_model.dropna(subset=FEATURES)

X = df_model[FEATURES]
y = df_model[TARGET]

print(f"Dataset para modelado: {X.shape[0]} filas, {X.shape[1]} columnas")

# ============================
# 3. Train / Test split
# ============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(f"Train: {X_train.shape[0]} filas | Test: {X_test.shape[0]} filas")

# ============================
# 4. Preprocesamiento
# ============================

numeric_transformer = Pipeline(
    steps=[
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ]
)

# ============================
# 5. Definir modelos
# ============================

models = {
    "ridge": Ridge(alpha=1.0, random_state=42),
    "random_forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ),
}

results = {}
best_model_name = None
best_mae = np.inf  # mientras más bajo, mejor
best_pipeline = None
best_y_pred = None

# ============================
# 6. Entrenar y evaluar
# ============================

for name, reg in models.items():
    print(f"\nEntrenando modelo: {name} ...")
    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", reg),
        ]
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {
        "MAE": float(mae),
        "R2": float(r2),
    }

    print(f"Resultados {name} -> MAE: {mae:,.0f} | R²: {r2:.3f}")

    if mae < best_mae:
        best_mae = mae
        best_model_name = name
        best_pipeline = pipe
        best_y_pred = y_pred

print("\n======================================")
print(f"Mejor modelo: {best_model_name} con MAE = {best_mae:,.0f}")
print("======================================")

# ============================
# 7. Guardar mejor modelo y artefactos
# ============================

MODEL_PATH = MODELS_DIR / "melb_model.pkl"
METRICS_PATH = MODELS_DIR / "melb_metrics.json"
PRED_PATH = MODELS_DIR / "melb_test_predictions.csv"

print(f"Guardando mejor modelo en: {MODEL_PATH}")
dump(best_pipeline, MODEL_PATH, compress=3)

print(f"Guardando métricas en: {METRICS_PATH}")
with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(
        {
            "mejor_modelo": best_model_name,
            "resultados": results,
        },
        f,
        indent=4,
        ensure_ascii=False,
    )

print(f"Guardando predicciones de test en: {PRED_PATH}")
df_pred = X_test.copy()
df_pred["Price_real"] = y_test
df_pred["Price_pred"] = best_y_pred
df_pred["Error"] = df_pred["Price_real"] - df_pred["Price_pred"]
df_pred.to_csv(PRED_PATH, index=False)

print("Entrenamiento completado y artefactos guardados correctamente.")
