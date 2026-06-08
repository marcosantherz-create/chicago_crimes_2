import streamlit as st
import pandas as pd
import numpy as np
import io
import pickle
import altair as alt
from google.cloud import storage
import river
from river import forest, metrics


# =========================================================
# CONFIGURACIÓN DE PÁGINA
# =========================================================
st.set_page_config(
    page_title="Predicción de Arrestos - Chicago",
    page_icon="👮🏽‍♂️",
    layout="wide"
)

# Paleta UP
UP_GREEN = "#055a4d"
UP_GRAY = "#a8a19f"
UP_WINE = "#8c203d"
UP_GOLD = "#c8a568"

st.title("👮🏽‍♂️ Aprendizaje en Línea: Crímenes de Chicago")
st.caption("Adaptive Random Forest • River ML • Google Cloud Storage • Cloud Run")

with st.expander("📖 Descripción técnica del proyecto"):
    st.markdown("""
    Este panel interactivo entrena un modelo **Adaptive Random Forest Classifier**
    de forma incremental, procesando datos históricos de crímenes de Chicago desde
    **Google Cloud Storage (GCS)**.

    **Enfoque Online:** por cada registro, el modelo primero predice si habrá un arresto
    (`Predict`) y posteriormente aprende de la realidad (`Learn`).

    Los datos provienen del sistema CLEAR del Departamento de Policía de Chicago y están
    disponibles públicamente mediante el Chicago Data Portal y BigQuery Public Datasets.
    """)


# =========================================================
# FUNCIONES AUXILIARES GCS
# =========================================================
def save_model_to_gcs(model_to_save, bucket_name, destination_blob):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob)

        with blob.open("wb", content_type="application/octet-stream") as f:
            pickle.dump(model_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"✅ Checkpoint guardado en gs://{bucket_name}/{destination_blob}")
        return True

    except Exception as e:
        print(f"❌ Error al guardar modelo en GCS: {e}")
        return False


def load_model_from_gcs(bucket_name, source_blob):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob)

        if blob.exists():
            data = blob.download_as_bytes()
            st.info("Modelo incremental recuperado desde GCS.")
            return pickle.loads(data)

        return None

    except Exception as e:
        st.warning(f"No se pudo cargar el modelo previo: {e}")
        return None


def delete_model_from_gcs(bucket_name, source_blob):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob)

        if blob.exists():
            blob.delete()
            st.success("Modelo eliminado de GCS.")
        else:
            st.info("No se encontró ningún modelo guardado en GCS.")

    except Exception as e:
        st.warning(f"Error al eliminar el modelo: {e}")


# =========================================================
# MODELO
# =========================================================
def new_model():
    return forest.ARFClassifier(n_models=10, seed=42)


# =========================================================
# SIDEBAR
# =========================================================
MODEL_PATH = "models/model_incremental.pkl"

st.sidebar.header("⚙️ Configuración")

with st.sidebar.expander("📂 Datos de entrada", expanded=True):
    bucket_name = st.text_input(
        "Bucket GCS",
        "ml_grandesdatos_proyectofinal"
    )

    prefix = st.text_input(
        "Prefijo / Carpeta",
        "chicago_crime_2/"
    )

    limite = st.number_input(
        "Muestras por archivo",
        value=5000,
        step=500
    )

    chunksize = st.number_input(
        "Tamaño del chunk",
        value=5000,
        step=500
    )

st.sidebar.markdown("---")
st.sidebar.subheader("💾 Persistencia del Modelo")


# =========================================================
# SESSION STATE
# =========================================================
if "model" not in st.session_state:
    try:
        loaded_model = load_model_from_gcs(bucket_name, MODEL_PATH)
    except Exception:
        loaded_model = None

    if loaded_model is None:
        loaded_model = new_model()

    st.session_state.model = loaded_model
    st.session_state.metric_acc = metrics.Accuracy()
    st.session_state.metric_prec = metrics.Precision()
    st.session_state.metric_rec = metrics.Recall()

    st.session_state.history = []
    st.session_state.history_file = []
    st.session_state.processed_files = []

    st.session_state.blobs = None
    st.session_state.index = 0
    st.session_state.auto_pilot_remaining = 0


if st.sidebar.button("💾 Guardar Checkpoint en GCS"):
    with st.spinner("Subiendo modelo a GCS..."):
        success = save_model_to_gcs(
            st.session_state.model,
            bucket_name,
            MODEL_PATH
        )

    if success:
        st.sidebar.success("Checkpoint guardado correctamente.")
    else:
        st.sidebar.error("Error al guardar en GCS. Revisa los logs.")


if st.sidebar.button("🔄 Reiniciar Modelo y Borrar Historial"):
    delete_model_from_gcs(bucket_name, MODEL_PATH)

    st.session_state.model = new_model()
    st.session_state.metric_acc = metrics.Accuracy()
    st.session_state.metric_prec = metrics.Precision()
    st.session_state.metric_rec = metrics.Recall()

    st.session_state.history = []
    st.session_state.history_file = []
    st.session_state.processed_files = []

    st.session_state.blobs = None
    st.session_state.index = 0
    st.session_state.auto_pilot_remaining = 0

    st.rerun()


model = st.session_state.model
metric_acc = st.session_state.metric_acc
metric_prec = st.session_state.metric_prec
metric_rec = st.session_state.metric_rec


# =========================================================
# PREPROCESAMIENTO
# =========================================================
def preprocess_chunk(df_chunk):
    df_chunk = df_chunk.copy()

    df_chunk["latitude"] = pd.to_numeric(df_chunk["latitude"], errors="coerce")
    df_chunk["longitude"] = pd.to_numeric(df_chunk["longitude"], errors="coerce")

    df_chunk = df_chunk.dropna(
        subset=["latitude", "longitude", "domestic", "district"]
    ).copy()

    min_latitude, max_latitude = 41.644, 42.023
    min_longitude, max_longitude = -87.940, -87.524

    df_chunk = df_chunk[
        (df_chunk["latitude"] >= min_latitude) &
        (df_chunk["latitude"] <= max_latitude) &
        (df_chunk["longitude"] >= min_longitude) &
        (df_chunk["longitude"] <= max_longitude)
    ].copy()

    df_chunk["date"] = pd.to_datetime(
        df_chunk["date"],
        utc=True,
        errors="coerce"
    )

    df_chunk = df_chunk.dropna(subset=["date"]).copy()

    df_chunk["day_of_week"] = df_chunk["date"].dt.day_name()
    df_chunk["month"] = df_chunk["date"].dt.month_name()
    df_chunk["hour_of_day"] = df_chunk["date"].dt.hour.astype(float)
    df_chunk["is_weekend"] = df_chunk["day_of_week"].isin(["Saturday", "Sunday"])

    df_chunk["district"] = df_chunk["district"].astype(str)

    features = [
        "primary_type",
        "location_description",
        "domestic",
        "district",
        "latitude",
        "longitude",
        "day_of_week",
        "month",
        "hour_of_day",
        "is_weekend"
    ]

    return df_chunk[features], df_chunk["arrest"]


# =========================================================
# ENTRENAMIENTO ONLINE
# =========================================================
def process_single_blob(bucket_name, blob_name, limite, chunksize):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    expected_raw_cols = {
        "date",
        "primary_type",
        "location_description",
        "arrest",
        "domestic",
        "district",
        "latitude",
        "longitude"
    }

    count = 0

    file_acc = metrics.Accuracy()
    file_prec = metrics.Precision()
    file_rec = metrics.Recall()

    try:
        content = blob.download_as_bytes()
        buffer = io.BytesIO(content)

        for chunk in pd.read_csv(
            buffer,
            chunksize=chunksize,
            low_memory=False,
            dtype={
                "latitude": str,
                "longitude": str,
                "district": str
            }
        ):
            if not expected_raw_cols.issubset(set(chunk.columns)):
                continue

            X_chunk, y_chunk = preprocess_chunk(chunk)

            if X_chunk.empty:
                continue

            for idx, row in X_chunk.iterrows():
                if count >= limite:
                    break

                try:
                    x = row.to_dict()
                    x["domestic"] = bool(x["domestic"])
                    x["is_weekend"] = bool(x["is_weekend"])

                    y = bool(y_chunk.loc[idx])

                    y_pred = model.predict_one(x)
                    if y_pred is None:
                        y_pred = False

                    model.learn_one(x, y)

                    metric_acc.update(y, y_pred)
                    metric_prec.update(y, y_pred)
                    metric_rec.update(y, y_pred)

                    file_acc.update(y, y_pred)
                    file_prec.update(y, y_pred)
                    file_rec.update(y, y_pred)

                    count += 1

                except Exception:
                    continue

            if count >= limite:
                break

        if count == 0:
            return None

        return {
            "count": count,
            "file_acc": file_acc.get(),
            "file_prec": file_prec.get(),
            "file_rec": file_rec.get(),
            "global_acc": metric_acc.get(),
            "global_prec": metric_prec.get(),
            "global_rec": metric_rec.get()
        }

    except Exception as e:
        st.error(f"Error procesando `{blob_name}`: {e}")
        return None


# =========================================================
# CONTROLADOR
# =========================================================
st.subheader("🚦 Flujo de Simulación en Tiempo Real")

if st.session_state.blobs is None:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    blobs = [b for b in blobs if b.name.lower().endswith(".csv")]
    st.session_state.blobs = blobs
    st.session_state.index = 0

if "auto_pilot_remaining" not in st.session_state:
    st.session_state.auto_pilot_remaining = 0

blobs = st.session_state.blobs
idx = st.session_state.index
total_archivos = len(blobs)

if total_archivos == 0:
    st.warning("No se encontraron archivos CSV con el prefijo configurado.")
    st.stop()


# =========================================================
# RESUMEN OPERATIVO
# =========================================================
estado = "🟢 En entrenamiento" if idx < total_archivos else "✅ Completado"

st.info(
    f"""
    **Estado:** {estado}  
    **Modelo:** Adaptive Random Forest  
    **Origen:** Google Cloud Storage  
    **Archivos procesados:** {idx}/{total_archivos}
    """
)

progress_value = min(idx / total_archivos, 1.0)
st.progress(progress_value)
st.caption(f"Progreso global: {idx} de {total_archivos} archivos procesados")


# =========================================================
# ACCIONES
# =========================================================
btn_col1, btn_col2 = st.columns(2)

with btn_col1:
    disabled_btn = st.session_state.auto_pilot_remaining > 0

    if st.button("🚀 Procesar Archivo", disabled=disabled_btn, use_container_width=True):
        if idx >= total_archivos:
            st.success("Todos los lotes de crímenes han sido procesados.")
        else:
            blob = blobs[idx]
            short_name = blob.name.split("/")[-1]

            st.info(f"Procesando lote {idx + 1}/{total_archivos}: `{short_name}`")

            with st.spinner("Entrenando ARFClassifier..."):
                result = process_single_blob(
                    bucket_name,
                    blob.name,
                    int(limite),
                    int(chunksize)
                )

            if result is not None:
                st.session_state.history.append(
                    (
                        result["global_acc"],
                        result["global_prec"],
                        result["global_rec"]
                    )
                )

                st.session_state.history_file.append(
                    (
                        result["file_acc"],
                        result["file_prec"],
                        result["file_rec"]
                    )
                )

                st.session_state.processed_files.append(short_name)

            st.session_state.index += 1
            st.rerun()


with btn_col2:
    if st.session_state.auto_pilot_remaining == 0:
        num_bloque = st.number_input(
            "Archivos a procesar en ráfaga",
            value=50,
            min_value=1,
            max_value=100,
            step=10
        )

        if st.button("🤖 Procesar Lotes Automáticamente", use_container_width=True):
            archivos_restantes = total_archivos - idx
            st.session_state.auto_pilot_remaining = min(
                int(num_bloque),
                archivos_restantes
            )
            st.rerun()

    else:
        if st.button("🛑 Detener Piloto Automático", use_container_width=True):
            st.session_state.auto_pilot_remaining = 0
            st.warning("Piloto automático detenido por el usuario.")
            st.rerun()


# =========================================================
# PILOTO AUTOMÁTICO
# =========================================================
if st.session_state.auto_pilot_remaining > 0:
    if idx >= total_archivos:
        st.session_state.auto_pilot_remaining = 0
        st.success("Todos los archivos del bucket han sido procesados.")

    else:
        blob = blobs[idx]
        short_name = blob.name.split("/")[-1]

        st.warning(
            f"🤖 Modo Piloto Automático Activo. "
            f"Quedan {st.session_state.auto_pilot_remaining} archivos."
        )

        st.info(f"Procesando: `{short_name}` ({idx + 1}/{total_archivos})")

        result = process_single_blob(
            bucket_name,
            blob.name,
            int(limite),
            int(chunksize)
        )

        if result is not None:
            st.session_state.history.append(
                (
                    result["global_acc"],
                    result["global_prec"],
                    result["global_rec"]
                )
            )

            st.session_state.history_file.append(
                (
                    result["file_acc"],
                    result["file_prec"],
                    result["file_rec"]
                )
            )

            st.session_state.processed_files.append(short_name)

        st.session_state.auto_pilot_remaining -= 1
        st.session_state.index += 1

        st.rerun()


# =========================================================
# KPIs
# =========================================================
st.markdown("---")
st.subheader("📌 Indicadores del Modelo")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Archivos Procesados",
    f"{st.session_state.index}/{total_archivos}"
)

col2.metric(
    "Accuracy",
    f"{metric_acc.get():.2%}"
)

col3.metric(
    "Precision",
    f"{metric_prec.get():.2%}"
)

col4.metric(
    "Recall",
    f"{metric_rec.get():.2%}"
)


# =========================================================
# GRÁFICAS
# =========================================================
if st.session_state.history and len(st.session_state.processed_files) > 0:

    accuracies_global = [h[0] for h in st.session_state.history]
    precisions_global = [h[1] for h in st.session_state.history]
    recalls_global = [h[2] for h in st.session_state.history]

    accuracies_local = [h[0] for h in st.session_state.history_file]
    precisions_local = [h[1] for h in st.session_state.history_file]
    recalls_local = [h[2] for h in st.session_state.history_file]

    min_len = min(
        len(st.session_state.processed_files),
        len(st.session_state.history),
        len(st.session_state.history_file)
    )

    df_metrics = pd.DataFrame({
        "Lote/Archivo": st.session_state.processed_files[:min_len],
        "Accuracy (Global)": accuracies_global[:min_len],
        "Precision (Global)": precisions_global[:min_len],
        "Recall (Global)": recalls_global[:min_len],
        "Accuracy (Este Lote)": accuracies_local[:min_len],
        "Precision (Este Lote)": precisions_local[:min_len],
        "Recall (Este Lote)": recalls_local[:min_len],
    })

    st.markdown("---")
    st.subheader("📈 Monitoreo de Aprendizaje Continuo")

    tab1, tab2, tab3 = st.tabs([
        "📈 Evolución Global",
        "🎯 Desempeño por Archivo",
        "📋 Datos Procesados"
    ])

    with tab1:
        st.markdown("#### Evolución de Métricas Globales Acumuladas")

        df_global = df_metrics.melt(
            id_vars=["Lote/Archivo"],
            value_vars=[
                "Accuracy (Global)",
                "Precision (Global)",
                "Recall (Global)"
            ],
            var_name="Métrica",
            value_name="Valor"
        )

        chart_global = (
            alt.Chart(df_global)
            .mark_line(
                point={"filled": True, "size": 100},
                strokeWidth=4
            )
            .encode(
                x=alt.X(
                    "Lote/Archivo:N",
                    sort=None,
                    title="Archivo Procesado",
                    axis=alt.Axis(labelAngle=-90)
                ),
                y=alt.Y(
                    "Valor:Q",
                    title="Valor",
                    scale=alt.Scale(domain=[0, 1])
                ),
                color=alt.Color(
                    "Métrica:N",
                    scale=alt.Scale(
                        domain=[
                            "Accuracy (Global)",
                            "Precision (Global)",
                            "Recall (Global)"
                        ],
                        range=[
                            UP_GREEN,
                            UP_WINE,
                            UP_GOLD
                        ]
                    ),
                    legend=alt.Legend(orient="bottom", title=None)
                ),
                tooltip=[
                    "Lote/Archivo",
                    "Métrica",
                    alt.Tooltip("Valor:Q", format=".4f")
                ]
            )
            .properties(height=500)
            .interactive()
        )

        st.altair_chart(chart_global, use_container_width=True)

        st.markdown("### Resumen actual")

        r1, r2, r3 = st.columns(3)

        r1.metric("Accuracy Máxima", f"{max(accuracies_global):.2%}")
        r2.metric("Precision Máxima", f"{max(precisions_global):.2%}")
        r3.metric("Recall Máximo", f"{max(recalls_global):.2%}")

    with tab2:
        st.markdown("#### Comportamiento del Modelo por Archivo")

        df_local = df_metrics.melt(
            id_vars=["Lote/Archivo"],
            value_vars=[
                "Accuracy (Este Lote)",
                "Precision (Este Lote)",
                "Recall (Este Lote)"
            ],
            var_name="Métrica",
            value_name="Valor"
        )

        chart_local = (
            alt.Chart(df_local)
            .mark_line(
                point={"filled": True, "size": 100},
                strokeWidth=4
            )
            .encode(
                x=alt.X(
                    "Lote/Archivo:N",
                    sort=None,
                    title="Archivo Procesado",
                    axis=alt.Axis(labelAngle=-90)
                ),
                y=alt.Y(
                    "Valor:Q",
                    title="Valor",
                    scale=alt.Scale(domain=[0, 1])
                ),
                color=alt.Color(
                    "Métrica:N",
                    scale=alt.Scale(
                        domain=[
                            "Accuracy (Este Lote)",
                            "Precision (Este Lote)",
                            "Recall (Este Lote)"
                        ],
                        range=[
                            UP_GREEN,
                            UP_WINE,
                            UP_GOLD
                        ]
                    ),
                    legend=alt.Legend(orient="bottom", title=None)
                ),
                tooltip=[
                    "Lote/Archivo",
                    "Métrica",
                    alt.Tooltip("Valor:Q", format=".4f")
                ]
            )
            .properties(height=500)
            .interactive()
        )

        st.altair_chart(chart_local, use_container_width=True)

    with tab3:
        st.dataframe(
            df_metrics.style.format({
                "Accuracy (Global)": "{:.4f}",
                "Precision (Global)": "{:.4f}",
                "Recall (Global)": "{:.4f}",
                "Accuracy (Este Lote)": "{:.4f}",
                "Precision (Este Lote)": "{:.4f}",
                "Recall (Este Lote)": "{:.4f}",
            }),
            use_container_width=True
        )

else:
    st.markdown("---")
    st.info(
        "💡 Las gráficas aparecerán aquí en tiempo real cuando proceses el primer archivo CSV."
    )


st.caption("Ecosistema: Cloud Run • River ML • Google Cloud Storage • Chicago Crime Dataset Abierto")
