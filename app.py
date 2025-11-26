# ================================================================
# app.py — Dashboard Melbourne (Académico + Técnico) en Dash
# Estilo: Glassmorphism moderno (assets/styles.css)
# Navegación: Sidebar vertical colapsable (full width)
# ================================================================

import functools
from pathlib import Path
import json

import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Folium
import folium
from folium.plugins import MarkerCluster


# ================================================================
# 1) RUTAS DE DATOS / MODELO (SIN CARGARLOS AÚN)
# ================================================================
DATA_PATH = Path("data/melb_data.csv")
MODEL_PATH = Path("models/melb_model.pkl")
METRICS_PATH = Path("models/melb_metrics.json")
PRED_PATH = Path("models/melb_test_predictions.csv")
ASSETS_MAP_PATH = Path("assets/melbourne_map.html")


# ================================================================
# 1.1) FUNCIONES DE CARGA PEREZOSA (LAZY LOADING)
# ================================================================

@functools.lru_cache(maxsize=1)
def load_dataframes():
    """Carga melb_data.csv una sola vez y devuelve (df_raw, df_copia)."""
    df_raw = pd.read_csv(DATA_PATH)
    df = df_raw.copy()
    return df_raw, df


@functools.lru_cache(maxsize=1)
def load_model():
    """Carga el modelo entrenado una sola vez."""
    return joblib.load(MODEL_PATH)


@functools.lru_cache(maxsize=1)
def load_metrics():
    """Carga el JSON de métricas una sola vez."""
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@functools.lru_cache(maxsize=1)
def load_pred_df():
    """Carga las predicciones y normaliza nombres de columnas una sola vez."""
    df_pred = pd.read_csv(PRED_PATH)

    rename_map = {}
    if "real_price" in df_pred.columns and "Price_real" not in df_pred.columns:
        rename_map["real_price"] = "Price_real"
    if "prediction" in df_pred.columns and "Price_pred" not in df_pred.columns:
        rename_map["prediction"] = "Price_pred"
    if "pred_price" in df_pred.columns and "Price_pred" not in df_pred.columns:
        rename_map["pred_price"] = "Price_pred"
    if "error" in df_pred.columns and "Error" not in df_pred.columns:
        rename_map["error"] = "Error"

    if rename_map:
        df_pred = df_pred.rename(columns=rename_map)

    return df_pred


# ================================================================
# 1.2) FEATURE IMPORTANCES Y MAPA — TAMBIÉN LAZY
# ================================================================

def _safe_extract_importances(model, df_sample):
    try:
        estimator = model
        # pipeline -> último estimador
        if hasattr(model, "steps"):
            estimator = model.steps[-1][1]
        # Buscar importances
        if not hasattr(estimator, "feature_importances_"):
            return None
        importances = estimator.feature_importances_
        # intentar obtener nombres de columnas
        names = None
        if hasattr(model, "feature_names_in_"):
            names = list(model.feature_names_in_)
        elif hasattr(estimator, "feature_names_in_"):
            names = list(estimator.feature_names_in_)
        else:
            names = df_sample.select_dtypes(include="number").columns.tolist()
        if len(names) != len(importances):
            min_len = min(len(names), len(importances))
            names = names[:min_len]
            importances = importances[:min_len]
        df_imp = pd.DataFrame({"Variable": names, "Importancia": importances}).sort_values(
            "Importancia", ascending=False
        )
        if df_imp.empty:
            return None
        fig = px.bar(
            df_imp.head(20),
            x="Importancia", y="Variable",
            orientation="h",
            title="Importancia de Variables (Top 20)",
            color="Importancia",
            color_continuous_scale="Blues",
            height=600,
            template="simple_white"
        )
        return fig
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def get_feature_importance_figure():
    """Calcula la figura de importancias una sola vez."""
    _, df = load_dataframes()
    model = load_model()
    sample_df = df.sample(min(1000, len(df))) if len(df) > 0 else df
    fig = _safe_extract_importances(model, sample_df)
    if fig is None:
        fig = go.Figure().update_layout(
            title="No fue posible obtener Feature Importance"
        )
    return fig


def ensure_map_created():
    """Crea/guarda el mapa de Folium solo si no existe y solo cuando se necesita."""
    if ASSETS_MAP_PATH.exists():
        return
    try:
        _, df = load_dataframes()
        m = folium.Map(location=[-37.8136, 144.9631], zoom_start=11, tiles="CartoDB positron")
        mc = MarkerCluster().add_to(m)
        lat_col = "Lattitude" if "Lattitude" in df.columns else None
        lon_col = "Longtitude" if "Longtitude" in df.columns else None
        if lat_col and lon_col:
            sample_points = df.dropna(subset=[lat_col, lon_col]).sample(min(500, len(df)))
            for _, row in sample_points.iterrows():
                folium.CircleMarker(
                    location=[row[lat_col], row[lon_col]],
                    radius=4, fill=True, fill_opacity=0.7,
                    popup=f"Price: {row['Price']:,.0f}"
                ).add_to(mc)
        m.save(ASSETS_MAP_PATH)
    except Exception:
        # no fallar el inicio de la app si algo pasa con folium
        pass


# ================================================================
# 2) APP INIT
# ================================================================
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    assets_folder="assets",
    suppress_callback_exceptions=True
)
server = app.server


# ================================================================
# 3) SIDEBAR (SECCIONES)
# ================================================================
sections = [
    ("intro", "1. Introducción"),
    ("contexto", "2. Contexto"),
    ("problema", "3. Planteamiento del Problema"),
    ("objetivos", "4. Objetivos y Justificación"),
    ("marco", "5. Marco Teórico"),
    ("metodo", "6. Metodología"),
    ("resultados", "7. Resultados y Análisis Final"),
    ("Almacenamiento", "7.1. Almacenamiento en la Nube"),
    ("conclusiones", "8. Conclusiones"),
]

sidebar = html.Div(
    id="sidebar",
    className="sidebar",
    children=[
        html.Div(
            [
                html.Button("«", id="toggle-btn", className="toggle-btn"),
                html.H2("Analisis Exploratorio de inmobilaria Melbourne", className="sidebar-title"),
            ],
            className="sidebar-header"
        ),
        html.Div(
            id="sidebar-items",
            children=[
                html.Div(
                    sec[1],
                    id=f"tab-{sec[0]}",
                    className="sidebar-item",
                    n_clicks=0
                ) for sec in sections
            ]
        )
    ]
)

content = html.Div(
    id="content",
    className="main-content",
    children=[html.Div("Selecciona una sección del menú", className="card")]
)

app.layout = html.Div(className="layout", children=[sidebar, content])


# ================================================================
# 4) SECCIONES ACADÉMICAS — VERSIONES LARGAS
# ================================================================
intro_md = """
Nuestro propósito general en este estudio es **comprender los factores que inciden en el precio de las propiedades**
(*variable objetivo: `Price`*), explorando cómo varía este valor según las características estructurales del inmueble
y su ubicación dentro de la ciudad. Para ello se analizan variables como:

- `Rooms`: número de habitaciones.  
- `Landsize`: superficie del terreno.  
- `BuildingArea`: superficie construida.  
- `Bathroom`: número de baños.  
- `Car`: número de plazas de parqueo.  
- `Distance`: distancia al distrito central de negocios (CBD).  
- `Type`, `Suburb`, `Regionname`: variables categóricas clave del mercado.

El proceso general del proyecto incluye:

1. **Exploración y análisis del dataset.**  
2. **Detección de valores faltantes y estrategia de imputación.**  
3. **Comparativa pre y post imputación.**  
4. **Modelado predictivo (regresión).**  
5. **Georreferenciación y visualización espacial de propiedades.**
6. **Almacenamiento en la Nube.**

Finalmente, se integran todos los resultados en un dashboard interactivo para la interpretación académica y técnica.
"""

contexto_md = """
El conjunto de datos **Melbourne Housing** recopila información de viviendas vendidas en distintas zonas de la ciudad,
incluyendo atributos físicos, ubicación y características del mercado.

Este dataset es ampliamente utilizado en analítica inmobiliaria porque permite entender:

- Cómo influyen variables estructurales (tamaño, habitaciones, baños).  
- El aporte de la localización y accesibilidad al centro de la ciudad.  
- Tendencias del mercado y patrones espaciales de valorización.  

Las variables espaciales (lat/long y suburbios) son clave para análisis geográfico y también enriquecen el modelo final,
ya que capturan parte del efecto del vecindario sobre el precio.
"""

problema_md = """
**Pregunta central:**  
¿Qué variables explican mejor el precio de las viviendas en Melbourne y qué tan bien podemos predecirlo?

Este problema se aborda mediante:

- diagnóstico de calidad de datos,  
- tratamiento de valores faltantes,  
- análisis estadístico y visual,  
- selección de modelo predictivo,  
- evaluación con métricas de regresión.

El desafío principal surge por la presencia de nulos en variables importantes y relaciones no lineales entre predictores
y la variable objetivo.
"""

objetivos_md = """
### Objetivo general
Desarrollar un modelo de regresión confiable para estimar el precio de viviendas en Melbourne,
integrando exploración de datos, imputación y modelado.

### Objetivos específicos
- Analizar estadísticos y distribuciones del dataset.  
- Identificar valores faltantes y definir estrategias de imputación.  
- Comparar el conjunto antes y después de imputar.  
- Entrenar y evaluar varios modelos de regresión.  
- Visualizar patrones espaciales de precios mediante mapas interactivos.  

### Justificación
El análisis inmobiliario soportado en datos permite mejorar decisiones de inversión,
planificación urbana y evaluación de riesgos financieros. Un modelo robusto aporta conocimiento sobre el mercado y puede
ser usado por compradores, vendedores y entidades financieras.
"""

marco_md = """
El proyecto se apoya en:

- **EDA (Exploración de datos):** análisis descriptivo, distribuciones univariadas, relaciones bivariadas y correlación.  
- **Imputación:** tratamiento de datos faltantes diferenciada por tipo de variable y contexto del mercado.  
- **Modelos de regresión:** comparativa de desempeño y selección del mejor.  
- **Evaluación:** MAE, RMSE y R² para medir capacidad predictiva.  
- **Análisis espacial:** georreferenciación de precios por zonas de la ciudad.

Estos componentes aseguran que el modelo final sea robusto, explicable y útil para un análisis académico completo.
"""


def cover_card():
    return html.Div(className="card cover-card", children=[
        html.H1("Melbourne Housing Dashboard", className="cover-title"),
        html.P("Análisis Exploratorio, Imputación, Modelado y Georreferenciación",
               className="cover-subtitle"),
        html.H4("Integrantes del proyecto"),
        html.Ul([
            html.Li("Juan Camilo Conrado"),
            html.Li("Juan Andrés Ramos"),
            html.Li("Sergio Cadavid"),
        ], className="cover-list"),
    ])


def content_intro():
    return html.Div(children=[
        cover_card(),
        html.Div(className="card", children=[
            html.Div(className="header-card", children=[
                html.H2("1. Introducción"),
            ]),
            dcc.Markdown(intro_md)
        ])
    ])


def content_contexto():
    return html.Div(className="card", children=[
        html.H2("2. Contexto"),
        dcc.Markdown(contexto_md)
    ])


def content_problema():
    return html.Div(className="card", children=[
        html.H2("3. Planteamiento del Problema"),
        dcc.Markdown(problema_md)
    ])


def content_objetivos():
    return html.Div(className="card", children=[
        html.H2("4. Objetivos y Justificación"),
        dcc.Markdown(objetivos_md)
    ])


def content_marco_teorico():
    return html.Div(className="card", children=[
        html.H2("5. Marco Teórico"),
        dcc.Markdown(marco_md)
    ])


# ================================================================
# 5) METODOLOGÍA — VERSIÓN LARGA
# ================================================================
metodo_intro_md = """
La metodología se desarrolló en etapas secuenciales. Primero se inspeccionó la calidad de los datos y sus distribuciones.
Luego se diseñó una estrategia de imputación consistente con la naturaleza de cada variable. Finalmente se entrenaron
modelos de regresión y se seleccionó el de mejor desempeño, incorporando análisis espacial mediante mapas interactivos.
"""


def metodo_definicion():
    return html.Div(className="card", children=[
        html.H2("a) Definición del problema"),
        dcc.Markdown("""
El problema se modela como una **regresión supervisada**.  
- **Variable objetivo:** `Price` (precio de la vivienda).  
- **Predictores:** variables numéricas y categóricas asociadas a características físicas y ubicación.

Se espera que la relación entre predictores y precio presente componentes no lineales y efectos espaciales.
        """),
    ])


def metodo_preparacion():
    df_raw, _ = load_dataframes()
    missing = df_raw.isnull().sum().reset_index()
    missing.columns = ["Variable", "Valores faltantes"]
    return html.Div(className="card", children=[
        html.H2("b) Preparación de datos"),
        dcc.Markdown("""
La preparación incluyó:
- Exploración de tipos de variables.
- Detección de valores faltantes.
- Revisión de coherencia (rangos, valores atípicos evidentes).
- Preparación para imputación y modelado.
        """),
        dbc.Table.from_dataframe(missing, striped=True, bordered=True, hover=True),
    ])


def metodo_imputacion():
    return html.Div(className="card", children=[
        html.H2("c) Imputación y transformaciones"),
        dcc.Markdown("""
Para lidiar con valores faltantes:
- **Categóricas:** imputación por moda/agrupación contextual (según suburbio, tipo, etc.).
- **Numéricas:** imputación por estadísticos o reglas basadas en grupos.

Tras imputar, se re-verificó la consistencia estadística para evitar sesgos fuertes en distribuciones.
        """),
    ])


def metodo_modelo():
    return html.Div(className="card", children=[
        html.H2("d) Selección del modelo"),
        dcc.Markdown("""
Se evaluaron modelos base de regresión (lineales y de ensamble).  
El **Random Forest Regressor** fue seleccionado por:
- Capturar relaciones no lineales.
- Robustez ante multicolinealidad.
- Buen desempeño con variables mixtas tras codificación.
        """),
    ])


def metodo_evaluacion():
    model_metrics = load_metrics()
    df_pred = load_pred_df()

    best_name = model_metrics.get("mejor_modelo", "random_forest")
    best_res = model_metrics.get("resultados", {}).get(best_name, {})
    mae = best_res.get("MAE", None)
    r2 = best_res.get("R2", None)
    rmse = None
    if "Error" in df_pred.columns:
        rmse = float((df_pred["Error"] ** 2).mean() ** 0.5)

    return html.Div(className="card", children=[
        html.H2("e) Evaluación del modelo"),
        dcc.Markdown("""
El desempeño se evaluó utilizando el conjunto de prueba.
- **MAE** cuantifica el error promedio absoluto.
- **RMSE** penaliza más los errores grandes.
- **R²** mide proporción de varianza explicada.
        """),
        html.Div(className="metric-card", children=[
            html.P(f"🧠 Modelo final: {best_name}"),
            html.P(f"📉 MAE: {mae:,.3f}" if mae is not None else "📉 MAE: —"),
            html.P(f"📊 RMSE: {rmse:,.3f}" if rmse is not None else "📊 RMSE: —"),
            html.P(f"📈 R²: {r2:.4f}" if r2 is not None else "📈 R²: —"),
        ])
    ])


# ================================================================
# 6) RESULTADOS D1 — EDA (LARGO + MÁS GRÁFICAS)
# ================================================================
def resultados_eda():

    df_raw, df = load_dataframes()

    eda_desc_md = """
### Análisis Exploratorio de Datos (EDA)

Se explora la variable objetivo `Price` y sus relaciones con los principales predictores.
El EDA permite identificar patrones generales del mercado, detectar posibles valores atípicos
y guiar las decisiones de imputación y selección de modelo.
    """

    faltantes_md = """
### Interpretación de valores faltantes

Se observan variables con porcentajes de nulos significativos. Estas columnas son relevantes para el modelo,
por lo que se definieron estrategias de imputación específicas para cada tipo de dato,
buscando preservar la estructura real del mercado.
    """

    fig_hist = px.histogram(df, x="Price", nbins=50,
                            title="Distribución del Precio (Price)",
                            template="simple_white")

    fig_dist = None
    if "Distance" in df.columns:
        fig_dist = px.scatter(df, x="Distance", y="Price",
                              title="Distance vs Price",
                              opacity=0.5,
                              template="simple_white")

    fig_type = None
    if "Type" in df.columns:
        fig_type = px.box(df, x="Type", y="Price",
                          title="Precio por Tipo de Vivienda (Type)",
                          template="simple_white")

    fig_rooms = None
    if "Rooms" in df.columns:
        fig_rooms = px.box(df, x="Rooms", y="Price",
                           title="Rooms vs Price",
                           template="simple_white")

    df_num = df.select_dtypes(include="number")
    corr = df_num.corr(numeric_only=True)
    fig_corr = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns, colorscale="Blues"
    ))
    fig_corr.update_layout(title="Matriz de Correlación Numérica", template="simple_white")

    return html.Div(children=[
        html.Div(className="card", children=[
            html.H2("a) EDA"),
            dcc.Markdown(eda_desc_md),
            dcc.Markdown(faltantes_md),
        ]),
        html.Div(className="card", children=[dcc.Graph(figure=fig_hist)]),
        html.Div(className="card", children=[
            dcc.Graph(figure=fig_dist) if fig_dist else html.P("No hay variable Distance.")
        ]),
        html.Div(className="card", children=[
            dcc.Graph(figure=fig_type) if fig_type else html.P("No hay variable Type.")
        ]),
        html.Div(className="card", children=[
            dcc.Graph(figure=fig_rooms) if fig_rooms else html.P("No hay variable Rooms.")
        ]),
        html.Div(className="card", children=[dcc.Graph(figure=fig_corr)]),
    ])


# ================================================================
# 8) RESULTADOS D3 — MODELO + MAPA (LARGO)
# ================================================================
def resultados_modelo():

    modelo_md = """
### Modelo Predictivo Final

El Random Forest Regressor fue elegido por su capacidad para modelar relaciones no lineales
y por generalizar mejor sobre datos heterogéneos. El desempeño obtenido evidencia una predicción
consistente del precio de viviendas.
    """

    geo_md = """
### Georreferenciación

El análisis espacial permite identificar zonas de alta valorización cercanas al CBD
y suburbios periféricos con menor costo relativo. El mapa interactivo refuerza la lectura
territorial de los precios.
    """

    model_metrics = load_metrics()
    df_pred = load_pred_df()

    best_name = model_metrics.get("mejor_modelo", "random_forest")
    best_res = model_metrics.get("resultados", {}).get(best_name, {})
    mae = best_res.get("MAE", None)
    r2 = best_res.get("R2", None)
    rmse = float((df_pred["Error"] ** 2).mean() ** 0.5) if "Error" in df_pred.columns else None

    # Figura de importancias (lazy)
    fig_imp = get_feature_importance_figure()

    # Limitar muestra para comparativa real vs predicho
    fig_comp = None
    if "Price_real" in df_pred.columns and "Price_pred" in df_pred.columns:
        df_comp_sample = df_pred.sample(min(1000, len(df_pred)))
        fig_comp = px.scatter(
            df_comp_sample, x="Price_real", y="Price_pred",
            title="Precio Real vs Predicho",
            opacity=0.6, template="simple_white"
        )
        fig_comp.add_shape(
            type="line",
            x0=df_comp_sample["Price_real"].min(),
            y0=df_comp_sample["Price_real"].min(),
            x1=df_comp_sample["Price_real"].max(),
            y1=df_comp_sample["Price_real"].max(),
            line=dict(color="red", width=2, dash="dash")
        )

    # Distribución del error con muestra limitada
    fig_err = None
    if "Error" in df_pred.columns:
        fig_err = px.histogram(
            df_pred.sample(min(2000, len(df_pred))), x="Error", nbins=40,
            title="Distribución del Error",
            template="simple_white"
        )

    # Generar mapa solo si es necesario
    ensure_map_created()
    if ASSETS_MAP_PATH.exists():
        map_frame = html.Iframe(
            src="/assets/melbourne_map.html",
            style={"width": "100%", "height": "600px", "border": "none", "border-radius": "18px"}
        )
    else:
        map_frame = html.Div("Mapa no disponible. Revisa assets/melbourne_map.html")

    return html.Div(children=[
        html.Div(className="card", children=[
            html.H2("c) Modelo Predictivo"),
            dcc.Markdown(modelo_md),
            html.H3(f"Modelo final: {best_name}"),
            html.Div(className="metric-card", children=[
                html.P(f"📉 MAE: {mae:,.3f}" if mae is not None else "📉 MAE: —"),
                html.P(f"📊 RMSE: {rmse:,.3f}" if rmse is not None else "📊 RMSE: —"),
                html.P(f"📈 R²: {r2:.4f}" if r2 is not None else "📈 R²: —"),
            ])
        ]),
        html.Div(className="card", children=[dcc.Graph(figure=fig_imp)]),
        html.Div(className="card", children=[
            dcc.Graph(figure=fig_comp) if fig_comp else html.P("No hay columnas Price_real/Price_pred.")
        ]),
        html.Div(className="card", children=[
            dcc.Graph(figure=fig_err) if fig_err else html.P("No hay columna Error.")
        ]),
        html.Div(className="card", children=[
            html.H3("Georreferenciación"),
            dcc.Markdown(geo_md)
        ]),
        html.Div(className="card", children=[
            html.H3("Mapa Interactivo de Melbourne"),
            map_frame
        ]),
    ])


# =============================================================
# 9) ALMACENAMIENTO EN LA NUBE – POSTGRESQL
# =============================================================

almacenamiento_md = """
El uso de **PostgreSQL como servicio en la nube** permitió centralizar, organizar 
y acceder de forma eficiente a la información inmobiliaria utilizada en este proyecto. 
Mediante una arquitectura cliente-servidor se garantizó que los datos estuvieran 
disponibles en cualquier momento, facilitando su consumo por las diferentes funciones 
de análisis, modelado y visualización.

La base de datos en PostgreSQL sirvió como un repositorio estructurado donde se 
almacenaron los datos originales, las transformaciones aplicadas y diferentes tablas 
derivadas del proceso analítico. Esto permitió mantener un **control de versiones**, 
evitar duplicados y garantizar la integridad de la información a lo largo del flujo 
de trabajo.

Además, la conexión a PostgreSQL permitió integrar el dashboard con un entorno 
de almacenamiento persistente y escalable, ideal para proyectos inmobiliarios gracias 
a su robustez, rendimiento y capacidad para manejar grandes volúmenes de registros.

En síntesis, el uso de PostgreSQL como backend del proyecto fortaleció la 
infraestructura analítica, habilitando un manejo profesional de los datos 
y facilitando su uso en futuras ampliaciones, análisis predictivos 
y despliegues productivos.
"""


def content_almacenamiento():
    return html.Div(className="card", children=[
        html.H2("Almacenamiento en la Nube – PostgreSQL"),
        dcc.Markdown(almacenamiento_md)
    ])


# ================================================================
# 10) CONCLUSIONES — LARGO
# ================================================================
conclusiones_md = """
El análisis del mercado inmobiliario de Melbourne permitió comprender las variables
que tienen mayor influencia en el precio final de las viviendas. A través de un proceso
estructurado de exploración, imputación, modelado y visualización, se obtuvieron
resultados consistentes.

El modelo **Random Forest Regressor** demostró ser adecuado para capturar relaciones no lineales,
alcanzando métricas favorables. La comparación pre y post imputación aseguró que la calidad
del dataset no se deteriorara, y el mapa espacial reforzó la lectura territorial de precios.

En síntesis, este proyecto evidencia la utilidad de la analítica de datos y machine learning
para estudios inmobiliarios, aportando herramientas interpretables para la toma de decisiones.
"""


def content_conclusiones():
    return html.Div(className="card", children=[
        html.H2("Conclusiones Finales"),
        dcc.Markdown(conclusiones_md)
    ])


# ================================================================
# 10) CALLBACK: COLAPSAR / EXPANDIR SIDEBAR
# ================================================================
@app.callback(
    Output("sidebar", "className"),
    Output("toggle-btn", "children"),
    Input("toggle-btn", "n_clicks"),
    prevent_initial_call=True
)
def toggle_sidebar(n):
    if n % 2 == 1:
        return "sidebar collapsed", "»"
    return "sidebar", "«"


# ================================================================
# 11) CALLBACK: NAVEGACIÓN GENERAL
# ================================================================
@app.callback(
    Output("content", "children"),
    [Input(f"tab-{s[0]}", "n_clicks") for s in sections],
    prevent_initial_call=True
)
def render_content(*args):

    clicked = [i for i, v in enumerate(args) if v]
    active = clicked[-1] if clicked else 0
    tab = sections[active][0]

    if tab == "intro":
        return content_intro()
    if tab == "contexto":
        return content_contexto()
    if tab == "problema":
        return content_problema()
    if tab == "objetivos":
        return content_objetivos()
    if tab == "marco":
        return content_marco_teorico()

    if tab == "metodo":
        return html.Div(children=[
            html.Div(className="card", children=[
                html.H2("6. Metodología"),
                dcc.Markdown(metodo_intro_md)
            ]),
            metodo_definicion(),
            metodo_preparacion(),
            metodo_imputacion(),
            metodo_modelo(),
            metodo_evaluacion()
        ])

    if tab == "resultados":
        return html.Div(children=[
            resultados_eda(),
            resultados_modelo()
        ])

    if tab == "Almacenamiento":
        return content_almacenamiento()

    if tab == "conclusiones":
        return content_conclusiones()

    return html.Div(className="card", children=[html.H2("Selecciona una sección del menú.")])


# ================================================================
# 12) RUN LOCAL
# ================================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)



