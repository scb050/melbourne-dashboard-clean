import pandas as pd
import numpy as np

# Paleta única (azules) para todo el informe
PALETTE = ["#cfe8ff", "#9dc9ff", "#6aa9ff", "#3a88f7", "#0f63d6"]
ACCENT  = "#0a3f8a"

RELEVANT_NUM = [
    "Price","Rooms","Distance","Landsize","BuildingArea",
    "Bedroom2","Bathroom","Car","Propertycount","YearBuilt"
]

KEY_COLS = set([
    "Price","Rooms","Distance","Landsize","BuildingArea",
    "Bedroom2","Bathroom","Car","Lattitude","Longtitude",
    "YearBuilt","CouncilArea","Regionname","Suburb","Postcode","Type"
])

def load_raw(path: str) -> pd.DataFrame:
    """Carga el CSV con tipado básico y limpieza mínima de nombres."""
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    # coerción numérica segura en columnas típicas
    for c in [c for c in RELEVANT_NUM if c in df.columns]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    return df

def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    pct = (df.isna().mean()*100).round(2)
    out = (pd.DataFrame({"variable": pct.index, "pct_missing": pct.values})
           .sort_values("pct_missing", ascending=False, ignore_index=True))
    return out

def skew_stat(s: pd.Series) -> float:
    return s.dropna().skew() if s.notna().any() else 0.0

def imputation_plan(df: pd.DataFrame) -> pd.DataFrame:
    """Regla por variable, según % faltantes y tipo (siguiendo rúbrica)."""
    pct = (df.isna().mean()*100)
    num_cols = df.select_dtypes(include=[np.number]).columns
    plan = []
    for col in df.columns:
        p = float(pct[col])
        tipo = "numérico" if col in num_cols else "categórico"
        if p == 0:
            regla = "Sin imputación."
        elif p <= 5:
            if tipo == "numérico":
                regla = f"Simple: {'mediana' if abs(skew_stat(df[col]))>1 else 'media'} (según asimetría)."
            else:
                regla = "Simple: moda global."
        elif p <= 30:
            regla = "Por grupos: Suburb (si existe), en su defecto Regionname."
        else:
            regla = ("Clave→ por grupos (Suburb/Regionname); No clave→ evaluar eliminación. "
                     "En este informe se conserva imputando de forma robusta.")
        plan.append((col, tipo, round(p,2), regla))
    return pd.DataFrame(plan, columns=["variable","tipo","% faltantes","decisión"])

def impute_df(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa según plan: 0–5% simple, 5–30% por grupos; >30% mantener si es clave."""
    out = df.copy()
    pct = (df.isna().mean()*100)
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = [c for c in df.columns if c not in num_cols]
    group_key = "Suburb" if "Suburb" in df.columns else ("Regionname" if "Regionname" in df.columns else None)

    # numéricos
    for col in num_cols:
        p = float(pct[col])
        if p == 0: 
            continue
        if p <= 5:
            fillv = out[col].median() if abs(skew_stat(out[col]))>1 else out[col].mean()
            out[col] = out[col].fillna(fillv)
        elif p <= 30:
            if group_key:
                out[col] = out.groupby(group_key)[col].transform(lambda s: s.fillna(s.median()))
            else:
                out[col] = out[col].fillna(out[col].median())
        else:
            if col in KEY_COLS:
                if group_key:
                    out[col] = out.groupby(group_key)[col].transform(lambda s: s.fillna(s.median()))
                else:
                    out[col] = out[col].fillna(out[col].median())
            else:
                out[col] = out[col].fillna(out[col].median())

    # categóricas
    for col in cat_cols:
        p = float(pct[col])
        if p == 0:
            continue
        if p <= 5:
            m = out[col].mode(dropna=True)
            out[col] = out[col].fillna(m.iloc[0] if not m.empty else "Desconocido")
        elif p <= 30:
            if group_key:
                def _fill(s):
                    m = s.mode(dropna=True)
                    return s.fillna(m.iloc[0] if not m.empty else "Desconocido")
                out[col] = out.groupby(group_key)[col].transform(_fill)
            else:
                m = out[col].mode(dropna=True)
                out[col] = out[col].fillna(m.iloc[0] if not m.empty else "Desconocido")
        else:
            if col in KEY_COLS:
                if group_key:
                    def _fill(s):
                        m = s.mode(dropna=True)
                        return s.fillna(m.iloc[0] if not m.empty else "Desconocido")
                    out[col] = out.groupby(group_key)[col].transform(_fill)
                else:
                    m = out[col].mode(dropna=True)
                    out[col] = out[col].fillna(m.iloc[0] if not m.empty else "Desconocido")
            else:
                out[col] = out[col].fillna("Desconocido")

    return out

def compare_distributions(df_before: pd.DataFrame, df_after: pd.DataFrame) -> pd.DataFrame:
    """Comparativa simple antes/después: media, mediana y desviación estándar por variable numérica."""
    num_cols = df_after.select_dtypes(include=[np.number]).columns
    rows = []
    for col in num_cols:
        b = df_before[col].dropna()
        a = df_after[col].dropna()
        rows.append({
            "variable": col,
            "media_antes": b.mean() if len(b)>0 else np.nan,
            "media_despues": a.mean() if len(a)>0 else np.nan,
            "mediana_antes": b.median() if len(b)>0 else np.nan,
            "mediana_despues": a.median() if len(a)>0 else np.nan,
            "std_antes": b.std(ddof=1) if len(b)>1 else np.nan,
            "std_despues": a.std(ddof=1) if len(a)>1 else np.nan,
        })
    comp = pd.DataFrame(rows)
    comp["delta_media"] = comp["media_despues"] - comp["media_antes"]
    comp["delta_mediana"] = comp["mediana_despues"] - comp["mediana_antes"]
    return comp
