import streamlit as st
import pandas as pd
import numpy as np
import io
import pickle

from google.cloud import storage
from river import linear_model, preprocessing, metrics, optim


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
st.set_page_config(
    page_title="Aprendizaje en línea",
    page_icon="🚕",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    padding-top: 3rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

.main-title {
    font-size: 2.7rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
}

.subtitle {
    font-size: 1.05rem;
    color: #d0d0d0;
    line-height: 1.6;
}

.info-box {
    background-color: #17324a;
    padding: 1rem;
    border-radius: 8px;
    margin: 1.5rem 0;
}

.footer {
    color: #999999;
    font-size: 0.85rem;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# FUNCIONES GCS
# =========================================================
def save_model_to_gcs(model, bucket_name, destination_blob):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob)
        blob.upload_from_string(pickle.dumps(model))
        st.success(f"Modelo guardado en GCS: `{destination_blob}`")
    except Exception as e:
        st.warning(f"No se pudo guardar el modelo: {e}")


def load_model_from_gcs(bucket_name, source_blob):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob)

        if blob.exists():
            data = blob.download_as_bytes()
            st.info("Modelo cargado desde GCS.")
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
            st.info("No había modelo guardado en GCS.")

    except Exception as e:
        st.warning(f"No se pudo eliminar el modelo: {e}")


# =========================================================
# MODELO
# =========================================================
def new_model():
    return preprocessing.StandardScaler() | linear_model.LinearRegression(
        optimizer=optim.SGD(0.001),
        intercept_lr=0.001
    )


MODEL_PATH = "models/model_incremental.pkl"


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("Configuración de Datos")

    bucket_name = st.text_input(
        "Bucket de GCS:",
        "bucket_131025"
    )

    prefix = st.text_input(
        "Prefijo/Carpeta:",
        "tlc_yellow_trips_2022/"
    )

    limite = st.number_input(
        "Filas a procesar por archivo:",
        value=1000,
        step=100,
        min_value=100
    )

    chunksize = st.number_input(
        "Tamaño del chunk de lectura:",
        value=500,
        step=100,
        min_value=100
    )

    st.divider()

    st.subheader("Persistencia del Modelo")

    save_checkpoint = st.button("💾 Guardar checkpoint en GCS")
    reset_button = st.button("🧹 Reiniciar modelo y borrar historial")


# =========================================================
# HEADER
# =========================================================
st.markdown(
    '<div class="main-title">Aprendizaje en Línea con River 🚕</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="subtitle">
Este panel interactivo entrena un modelo de <b>aprendizaje incremental</b>
con River, procesando archivos CSV desde <b>Google Cloud Storage</b>.
<br><br>
<b>Enfoque online:</b> por cada registro, el modelo primero predice y luego aprende de la realidad.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
Modelo incremental preparado para procesar datos desde GCS.
</div>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================
if "model" not in st.session_state:
    loaded_model = load_model_from_gcs(bucket_name, MODEL_PATH)

    if loaded_model is None:
        loaded_model = new_model()

    st.session_state.model = loaded_model
    st.session_state.metric_r2 = metrics.R2()
    st.session_state.metric_mae = metrics.MAE()

    st.session_state.history_r2 = []
    st.session_state.history_mae = []
    st.session_state.history_file_r2 = []
    st.session_state.history_file_mae = []
    st.session_state.processed_files = []

    st.session_state.blobs = None
    st.session_state.index = 0


model = st.session_state.model
metric_r2 = st.session_state.metric_r2
metric_mae = st.session_state.metric_mae


# =========================================================
# BOTONES SIDEBAR
# =========================================================
if save_checkpoint:
    save_model_to_gcs(model, bucket_name, MODEL_PATH)

if reset_button:
    delete_model_from_gcs(bucket_name, MODEL_PATH)

    st.session_state.model = new_model()
    st.session_state.metric_r2 = metrics.R2()
    st.session_state.metric_mae = metrics.MAE()

    st.session_state.history_r2 = []
    st.session_state.history_mae = []
    st.session_state.history_file_r2 = []
    st.session_state.history_file_mae = []
    st.session_state.processed_files = []

    st.session_state.blobs = None
    st.session_state.index = 0

    st.success("Entrenamiento reiniciado correctamente.")
    st.rerun()


# =========================================================
# FEATURE ENGINEERING
# =========================================================
def _parse_time_fields(row):
    if "pickup_hour" in row and pd.notna(row["pickup_hour"]):
        try:
            hour = int(pd.to_numeric(row["pickup_hour"], errors="coerce"))
            return None, max(0, min(hour, 23))
        except Exception:
            pass

    for c in ("tpep_pickup_datetime", "lpep_pickup_datetime", "pickup_datetime"):
        if c in row and pd.notna(row[c]):
            dt = pd.to_datetime(row[c], errors="coerce", utc=False)
            if pd.notna(dt):
                return dt, int(dt.hour)

    return None, 0


def _extract_x(row):
    dist = float(row["trip_distance"])
    psg = float(row["passenger_count"])

    dt, hour = _parse_time_fields(row)

    if isinstance(dt, pd.Timestamp):
        dow = int(dt.weekday())
    else:
        dow = 0

    weekend = 1.0 if dow >= 5 else 0.0

    return {
        "dist": dist,
        "log_dist": float(np.log1p(max(dist, 0))),
        "pass": psg,
        "hour": float(hour),
        "dow": float(dow),
        "is_weekend": weekend,
    }


# =========================================================
# PROCESAR ARCHIVO
# =========================================================
def process_single_blob(bucket_name, blob_name, limite=1000, chunksize=500):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    chunks_validos = []

    try:
        content = blob.download_as_bytes()
        buffer = io.BytesIO(content)

        for chunk in pd.read_csv(buffer, chunksize=chunksize, low_memory=False):
            cols_needed = ["trip_distance", "passenger_count", "fare_amount"]

            if not set(cols_needed).issubset(chunk.columns):
                continue

            for col in cols_needed:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

            chunk = chunk.replace([np.inf, -np.inf], np.nan)
            chunk = chunk.dropna(subset=cols_needed)

            chunk = chunk[
                chunk["fare_amount"].between(2, 200)
                & chunk["trip_distance"].between(0.1, 50)
                & chunk["passenger_count"].between(1, 6)
            ]

            if not chunk.empty:
                chunks_validos.append(chunk)

        if not chunks_validos:
            return None

        df_file = pd.concat(chunks_validos, ignore_index=True)

        if len(df_file) > limite:
            df_file = df_file.sample(n=limite, random_state=42)

        file_r2 = metrics.R2()
        file_mae = metrics.MAE()

        count = 0

        for _, row in df_file.iterrows():
            y = float(row["fare_amount"])
            x = _extract_x(row)

            pred = model.predict_one(x)

            if pred is None:
                pred_eval = 0.0
            else:
                pred_eval = float(np.clip(pred, 2, 200))

            metric_r2.update(y, pred_eval)
            metric_mae.update(y, pred_eval)

            file_r2.update(y, pred_eval)
            file_mae.update(y, pred_eval)

            model.learn_one(x, y)

            count += 1

    except Exception as e:
        st.warning(f"Error en `{blob_name}`: {e}")
        return None

    return {
        "count": count,
        "file_r2": file_r2.get(),
        "file_mae": file_mae.get(),
        "global_r2": metric_r2.get(),
        "global_mae": metric_mae.get()
    }


# =========================================================
# ESTADO ACTUAL
# =========================================================
st.divider()
st.subheader("Estado actual del modelo")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Archivo procesado actual",
        value=st.session_state.index
    )

with col2:
    st.metric(
        label="R² acumulado",
        value=f"{metric_r2.get():.4f}"
    )

with col3:
    st.metric(
        label="MAE acumulado",
        value=f"{metric_mae.get():.4f}"
    )


# =========================================================
# PROGRESO
# =========================================================
if st.session_state.blobs:
    total_files = len(st.session_state.blobs)
    current_file = st.session_state.index

    progress_value = current_file / total_files if total_files > 0 else 0

    st.progress(progress_value)
    st.caption(f"Progreso: {current_file}/{total_files} archivos procesados")


# =========================================================
# PROCESAMIENTO
# =========================================================
st.divider()
st.subheader("Procesamiento incremental")

col_btn, col_text = st.columns([1, 3])

with col_btn:
    process_button = st.button("Procesar siguiente archivo ➡️")

with col_text:
    st.caption(
        "Cada clic procesa un archivo CSV, actualiza métricas y entrena el modelo incrementalmente."
    )


if process_button:
    if st.session_state.blobs is None:
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        blobs = list(bucket.list_blobs(prefix=prefix))

        blobs = [
            b for b in blobs
            if b.name.endswith(".csv") and not b.name.endswith("/")
        ]

        st.session_state.blobs = blobs
        st.session_state.index = 0

        st.info(f"Se encontraron {len(blobs)} archivos CSV en `{prefix}`.")

    blobs = st.session_state.blobs
    idx = st.session_state.index

    if idx >= len(blobs):
        st.success("Todos los archivos ya fueron procesados.")

    else:
        blob = blobs[idx]
        short = blob.name.split("/")[-1]

        with st.spinner(f"Procesando archivo {idx + 1}/{len(blobs)}: {short}"):
            result = process_single_blob(
                bucket_name=bucket_name,
                blob_name=blob.name,
                limite=int(limite),
                chunksize=int(chunksize)
            )

        if result is not None:
            st.session_state.history_r2.append(result["global_r2"])
            st.session_state.history_mae.append(result["global_mae"])

            st.session_state.history_file_r2.append(result["file_r2"])
            st.session_state.history_file_mae.append(result["file_mae"])

            st.session_state.processed_files.append(short)

            st.success(f"Archivo procesado correctamente: `{short}`")

            res1, res2, res3 = st.columns(3)

            with res1:
                st.metric("Registros procesados", result["count"])

            with res2:
                st.metric("R² archivo", f"{result['file_r2']:.4f}")

            with res3:
                st.metric("MAE archivo", f"{result['file_mae']:.4f}")

            save_model_to_gcs(model, bucket_name, MODEL_PATH)

        else:
            st.warning("No se procesaron registros válidos en este archivo.")

        st.session_state.index += 1
        st.rerun()


# =========================================================
# HISTORIAL
# =========================================================
st.divider()

if st.session_state.history_r2:
    df_hist = pd.DataFrame({
        "archivo": st.session_state.processed_files,
        "R2_archivo": st.session_state.history_file_r2,
        "MAE_archivo": st.session_state.history_file_mae,
        "R2_acumulado": st.session_state.history_r2,
        "MAE_acumulado": st.session_state.history_mae,
    })

    st.subheader("Historial de procesamiento")
    st.dataframe(df_hist, use_container_width=True)

    st.subheader("Evolución de métricas acumuladas")
    st.line_chart(df_hist[["R2_acumulado", "MAE_acumulado"]])

    st.subheader("Métricas por archivo")
    st.line_chart(df_hist[["R2_archivo", "MAE_archivo"]])

else:
    st.info(
        "Las gráficas de evolución aparecerán aquí cuando proceses el primer archivo CSV."
    )


# =========================================================
# FOOTER
# =========================================================
st.markdown(
    '<div class="footer">Cloud Run + River ML • Dataset público de taxis NYC</div>',
    unsafe_allow_html=True
)
