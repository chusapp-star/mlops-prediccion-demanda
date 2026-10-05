import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "modelo_demanda_produccion.joblib"
META_FILE = BASE_DIR / "metadata_modelo.json"
MODELS_FILE = BASE_DIR / "Comparacion_Modelos_Demanda.csv"
IMPORTANCE_FILE = BASE_DIR / "Importancia_Variables.csv"
MONITOR_FILE = BASE_DIR / "Monitoreo_Modelo_Demanda.csv"
EXAMPLE_FILE = BASE_DIR / "datos_nuevos_ejemplo.csv"

st.set_page_config(
    page_title="Centro Inteligente de Demanda | MLOps",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
      .hero {padding: 1.5rem 1.7rem; border-radius: 18px; background: linear-gradient(135deg,#0f172a,#1e293b); color:white; margin-bottom:1rem;}
      .hero h1 {margin:0; font-size:2.05rem;}
      .hero p {margin:.45rem 0 0 0; color:#cbd5e1; font-size:1.02rem;}
      .status-ok {display:inline-block; padding:.35rem .65rem; border-radius:999px; background:#dcfce7; color:#166534; font-weight:700;}
      .status-watch {display:inline-block; padding:.35rem .65rem; border-radius:999px; background:#fef3c7; color:#92400e; font-weight:700;}
      .note {padding: .9rem 1rem; border-radius:12px; background:#f8fafc; border:1px solid #e2e8f0;}
      div[data-testid="stMetric"] {border:1px solid #e2e8f0; padding:14px; border-radius:14px; background:white;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_csv(path: Path):
    return pd.read_csv(path)


def prepare_input(df: pd.DataFrame) -> pd.DataFrame:
    """Acepta datos crudos del laboratorio o datos ya preparados."""
    data = df.copy()

    if "Fecha" in data.columns:
        fecha = pd.to_datetime(data["Fecha"], errors="coerce")
        if "Anio" not in data.columns:
            data["Anio"] = fecha.dt.year
        if "Numero_Dia_Semana" not in data.columns:
            data["Numero_Dia_Semana"] = fecha.dt.dayofweek
        if "Dia_Mes" not in data.columns:
            data["Dia_Mes"] = fecha.dt.day

    if "Descuento_Porc" in data.columns:
        data["Descuento_Porc"] = (
            data["Descuento_Porc"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.strip()
        )
        data["Descuento_Porc"] = pd.to_numeric(data["Descuento_Porc"], errors="coerce")

    cols_remove = [
        "ID_Registro",
        "Fecha",
        "Demanda_Dia_Siguiente",
        "Demanda_Predicha_Previa",
        "Error_Absoluto_Previa",
        "Version_Modelo",
        "Latencia_ms",
        "Drift_Score",
        "Alerta_Drift",
    ]
    data = data.drop(columns=cols_remove, errors="ignore")
    return data


metadata = load_json(META_FILE)
model = load_model()
model_results = load_csv(MODELS_FILE)
importance = load_csv(IMPORTANCE_FILE)
monitoring = load_csv(MONITOR_FILE)

mae = float(metadata["MAE"])
rmse = float(metadata["RMSE"])
r2 = float(metadata["R2"])
mae_warning = float(metadata.get("umbral_advertencia_mae", mae * 1.25))

st.markdown(
    """
    <div class="hero">
      <h1>Centro Inteligente de Predicción de Demanda</h1>
      <p>Modelo de Machine Learning llevado a una solución MLOps para apoyar decisiones de inventario, abastecimiento, promociones y planificación.</p>
      <p style="margin-top:.75rem; font-size:.92rem; color:#94a3b8;"><b>Estudiante:</b> Jesús Prado Portuguez · Laboratorio 4 · Analítica Predictiva y Aprendizaje Automático</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("MLOps Demand Control")
st.sidebar.caption("Jesús Prado Portuguez · Laboratorio 4 · Analítica Predictiva y Aprendizaje Automático")
section = st.sidebar.radio(
    "Navegación",
    [
        "Resumen ejecutivo",
        "Nueva predicción",
        "Desempeño del modelo",
        "Monitoreo MLOps",
        "Arquitectura y uso responsable",
    ],
)
st.sidebar.divider()
st.sidebar.write(f"**Modelo:** {metadata['nombre_modelo'].replace('_', ' ')}")
st.sidebar.write(f"**Versión:** {metadata['version']}")
st.sidebar.write(f"**Objetivo:** {metadata['variable_objetivo']}")
st.sidebar.caption("La predicción apoya la decisión empresarial; no sustituye el criterio profesional.")


if section == "Resumen ejecutivo":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", f"{mae:.2f} uds")
    c2.metric("RMSE", f"{rmse:.2f} uds")
    c3.metric("R²", f"{r2:.3f}")
    c4.metric("Registros de prueba", f"{metadata['numero_registros_prueba']:,}")

    historical_alert_rate = 100 * (monitoring.get("Error_Absoluto_Modelo", pd.Series(dtype=float)) > mae_warning).mean()
    if mae < mae_warning and r2 >= 0.75:
        status_html = '<span class="status-ok">OPERATIVO CON MONITOREO</span>'
    else:
        status_html = '<span class="status-watch">REVISAR MODELO</span>'

    st.markdown(f"### Estado de referencia del modelo {status_html}", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="note">
        <b>Lectura gerencial:</b> el modelo se desvía en promedio unas <b>{mae:.1f} unidades</b> respecto a la demanda real y explica aproximadamente <b>{r2*100:.1f}%</b> de la variación observada. 
        Para una operación real, esto permite planificar con bastante más criterio que usar únicamente un promedio histórico, pero sigue siendo necesario revisar los casos de error alto y cambios en el comportamiento de los datos.
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Comparación de modelos")
        fig = px.bar(
            model_results.sort_values("MAE", ascending=True),
            x="MAE",
            y="Modelo",
            orientation="h",
            text_auto=".2f",
            title="Menor MAE = menor error promedio",
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Variables con mayor aporte predictivo")
        top = importance.head(8).sort_values("Importancia", ascending=True)
        fig = px.bar(
            top,
            x="Importancia",
            y="Variable",
            orientation="h",
            title="Importancia por permutación",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Qué decisión puede apoyar")
    a, b, c = st.columns(3)
    a.info("**Inventario:** anticipar cuánto producto conviene tener disponible y reducir faltantes o sobreinventario.")
    b.info("**Promociones:** estimar si una campaña o descuento puede coincidir con una demanda mayor y preparar la operación.")
    c.info("**Planificación:** asignar recursos con base en demanda esperada, sin depender únicamente de intuición.")


elif section == "Nueva predicción":
    st.header("Predicción con nuevos datos")
    st.write(
        "La aplicación acepta un CSV con registros nuevos. Puede usar el archivo de ejemplo incluido en el proyecto o cargar datos con la misma estructura de la base del laboratorio."
    )

    with open(EXAMPLE_FILE, "rb") as f:
        st.download_button(
            "Descargar CSV de ejemplo",
            data=f.read(),
            file_name="datos_nuevos_ejemplo.csv",
            mime="text/csv",
        )

    uploaded = st.file_uploader("Seleccione un archivo CSV", type=["csv"])
    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        st.caption(f"Registros recibidos: {len(raw):,}")
        st.dataframe(raw.head(10), use_container_width=True)

        prepared = prepare_input(raw)
        expected = list(model.named_steps["preprocesamiento"].feature_names_in_)
        missing = [c for c in expected if c not in prepared.columns]

        if missing:
            st.error("Faltan columnas necesarias para ejecutar el modelo: " + ", ".join(missing))
        else:
            prepared = prepared[expected]
            if st.button("Generar predicciones", type="primary"):
                try:
                    preds = model.predict(prepared)
                    result = raw.copy()
                    result["Demanda_Predicha"] = np.maximum(preds, 0)
                    st.success("Predicciones generadas correctamente.")

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Demanda promedio prevista", f"{result['Demanda_Predicha'].mean():.1f}")
                    c2.metric("Demanda máxima prevista", f"{result['Demanda_Predicha'].max():.1f}")
                    c3.metric("Registros procesados", f"{len(result):,}")

                    st.dataframe(result, use_container_width=True)
                    fig = px.histogram(result, x="Demanda_Predicha", nbins=20, title="Distribución de la demanda predicha")
                    st.plotly_chart(fig, use_container_width=True)

                    csv = result.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "Descargar predicciones",
                        data=csv,
                        file_name="predicciones_demanda.csv",
                        mime="text/csv",
                    )
                except Exception as exc:
                    st.error(f"No fue posible realizar la predicción: {exc}")


elif section == "Desempeño del modelo":
    st.header("Desempeño y análisis de errores")
    st.dataframe(model_results.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R2": "{:.4f}"}), use_container_width=True)

    left, right = st.columns(2)
    with left:
        fig = px.scatter(
            monitoring,
            x="Demanda_Real",
            y="Demanda_Predicha_Modelo",
            hover_data=["Error_Absoluto_Modelo"],
            title="Demanda real vs. demanda predicha",
        )
        min_v = min(monitoring["Demanda_Real"].min(), monitoring["Demanda_Predicha_Modelo"].min())
        max_v = max(monitoring["Demanda_Real"].max(), monitoring["Demanda_Predicha_Modelo"].max())
        fig.add_shape(type="line", x0=min_v, y0=min_v, x1=max_v, y1=max_v, line=dict(dash="dash"))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.histogram(
            monitoring,
            x="Error_Absoluto_Modelo",
            nbins=30,
            title="Distribución del error absoluto",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Casos con mayor diferencia")
    cols = [c for c in ["Tienda", "Provincia", "Canal_Venta", "Categoria_Producto", "Tipo_Promocion", "Demanda_Real", "Demanda_Predicha_Modelo", "Error_Absoluto_Modelo"] if c in monitoring.columns]
    st.dataframe(
        monitoring.sort_values("Error_Absoluto_Modelo", ascending=False)[cols].head(15),
        use_container_width=True,
    )

    st.markdown(
        """
        **Interpretación:** los errores altos no significan automáticamente que el modelo sea inútil. Son precisamente los casos que deben revisarse para identificar promociones atípicas, cambios repentinos de demanda, categorías poco frecuentes o condiciones que no estuvieron suficientemente representadas durante el entrenamiento.
        """
    )


elif section == "Monitoreo MLOps":
    st.header("Monitoreo, alertas y reentrenamiento")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE de referencia", f"{mae:.2f}")
    c2.metric("Alerta MAE", f"> {mae_warning:.2f}")
    c3.metric("Error real promedio", f"{monitoring['Error_Absoluto_Modelo'].mean():.2f}")
    c4.metric("R² de referencia", f"{r2:.3f}")

    st.markdown(
        f"""
        <div class="note">
        <b>Criterio propuesto:</b> no reentrenaría el modelo por un único error aislado. Primero revisaría tendencia. 
        Una advertencia técnica se activa si el MAE supera aproximadamente <b>{mae_warning:.2f}</b> unidades (25% sobre la referencia). 
        Si el deterioro continúa en dos periodos consecutivos, si aparecen categorías o canales nuevos, o si existe drift persistente, entonces sí recomendaría reentrenar.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Error por segmento")
    dimension = st.selectbox(
        "Analizar MAE por",
        [c for c in ["Canal_Venta", "Categoria_Producto", "Tipo_Promocion", "Tienda", "Provincia", "Dia_Semana", "Feriado"] if c in monitoring.columns],
    )
    segment = (
        monitoring.groupby(dimension, dropna=False)
        .agg(MAE_Segmento=("Error_Absoluto_Modelo", "mean"), Registros=("Demanda_Real", "size"))
        .reset_index()
        .sort_values("MAE_Segmento", ascending=False)
    )
    fig = px.bar(segment, x=dimension, y="MAE_Segmento", text_auto=".2f", title=f"MAE por {dimension}")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Plan de mantenimiento mensual")
    st.markdown(
        """
        1. Comparar MAE, RMSE y R² del periodo con los valores de referencia del modelo.
        2. Revisar los segmentos con mayor error y los registros extremos.
        3. Validar cambios en canales, productos, promociones y distribución de variables.
        4. Revisar alertas de data drift y posibles cambios de comportamiento (concept drift).
        5. Registrar la decisión: mantener, investigar o reentrenar, dejando trazabilidad de la versión.
        """
    )


else:
    st.header("Arquitectura y uso responsable")
    st.markdown(
        """
        ### Flujo de la solución
        **Datos → Preparación → Entrenamiento → Evaluación → Modelo seleccionado → Serialización → Streamlit → Usuario empresarial → Predicción → Monitoreo → Alerta → Reentrenamiento**

        **Por qué importa:** guardar el pipeline completo evita que producción aplique transformaciones distintas a las utilizadas durante el entrenamiento. La aplicación convierte el modelo en una herramienta accesible, mientras que el monitoreo permite detectar cuándo la realidad empresarial cambia y el modelo deja de comportarse como se esperaba.
        """
    )

    st.subheader("Data drift vs. concept drift")
    c1, c2 = st.columns(2)
    c1.info(
        "**Data drift:** cambian las características de los datos de entrada. Por ejemplo, aumenta el peso del e-commerce, aparecen productos nuevos o cambia el nivel habitual de descuentos."
    )
    c2.info(
        "**Concept drift:** cambia la relación entre esas variables y la demanda. Por ejemplo, una promoción que antes elevaba ventas deja de generar el mismo efecto porque cambió el comportamiento del cliente."
    )

    st.subheader("Riesgos y controles")
    risks = pd.DataFrame(
        {
            "Riesgo": [
                "Data leakage",
                "Credenciales o datos sensibles en GitHub",
                "Modelo desactualizado",
                "Sesgo por datos históricos",
                "Uso automático sin criterio humano",
            ],
            "Control propuesto": [
                "Excluir variables posteriores a la predicción y métricas de modelos previos.",
                "Usar .gitignore, variables de entorno y no publicar información confidencial.",
                "Monitorear métricas y drift; reentrenar cuando exista deterioro sostenido.",
                "Revisar desempeño por segmentos y evitar interpretar importancia como causalidad.",
                "Usar la predicción como apoyo a decisiones, no como decisión final automática.",
            ],
        }
    )
    st.dataframe(risks, use_container_width=True, hide_index=True)
