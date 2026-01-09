"""
Dashboard de Ventas (Práctica Streamlit)

Este script construye un dashboard interactivo en Streamlit para analizar ventas.
Incluye 4 secciones (según el enunciado) y añade extras "executive-ready":
- Estilo visual (turquesa + naranja suave) y tipografía.
- Contexto en cada gráfico (micro-narrativa) para que se entienda rápido.
- Animación con Plotly (Top productos por año) -> activable (consume recursos).
- Comparador (Store A vs Store B) en la pestaña "Insights Extra".
- Carga robusta de datos desde ZIP (local en repo o desde URLs).

Cómo ejecutar en local (VS Code):
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import zipfile
import urllib.request
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ============================================================
# CONFIG / THEME
# ============================================================
ACCENT_TURQ = "#1FB6AA"
ACCENT_ORANGE = "#FFB168"
ACCENT_DARK = "#0B1320"

px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = [
    ACCENT_TURQ,
    ACCENT_ORANGE,
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#B279A2",
]

st.set_page_config(page_title="Dashboard Ventas", page_icon="📊", layout="wide")

# ============================================================
# DATA SOURCES (elige UNA estrategia: local zip en repo o URLs)
# ============================================================
# Opción A (la que tú has hecho ahora): ZIPs en el repo (en la raíz)
LOCAL_ZIP_1 = Path("parte_1.zip")
LOCAL_ZIP_2 = Path("parte_2.zip")

# Opción B: ZIPs en GitHub Releases (si prefieres NO tenerlos en el repo)
ZIP_URL_1 = "https://github.com/macgelado/practica_streamlilt/releases/download/v1/parte_1.zip"
ZIP_URL_2 = "https://github.com/macgelado/practica_streamlilt/releases/download/v1/parte_2.zip"

DATA_DIR = Path("data")
CACHE_DIR = Path(".cache")  # para no re-descargar ni re-extraer todo el rato


# ============================================================
# CSS (idea 10: look más web-app)
# ============================================================
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

/* "Más web": ocultar menú/footer */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}

html, body, [class*="css"] {{
  font-family: 'Inter', sans-serif !important;
}}

.block-container {{
  padding-top: 1.2rem;
  padding-bottom: 2rem;
}}

.small-note {{
  color: rgba(11, 19, 32, 0.65);
  font-size: 0.92rem;
}}

.badge {{
  display: inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(31, 182, 170, 0.12);
  color: {ACCENT_TURQ};
  font-weight: 700;
  font-size: 0.85rem;
}}

.banner {{
  background: linear-gradient(90deg, rgba(31,182,170,0.14), rgba(255,177,104,0.18));
  border: 1px solid rgba(11, 19, 32, 0.08);
  border-radius: 18px;
  padding: 14px 16px;
  margin-bottom: 14px;
}}

.banner-title {{
  font-size: 1.05rem;
  font-weight: 800;
  color: {ACCENT_DARK};
  margin-bottom: 4px;
}}

.banner-sub {{
  color: rgba(11, 19, 32, 0.70);
  font-size: 0.92rem;
}}

div[data-testid="stMetric"] {{
  background: rgba(31, 182, 170, 0.08);
  border: 1px solid rgba(31, 182, 170, 0.18);
  padding: 14px;
  border-radius: 16px;
}}
div[data-testid="stMetric"] [data-testid="stMetricLabel"] {{
  color: rgba(11, 19, 32, 0.70);
}}
hr {{
  border-top: 1px solid rgba(11, 19, 32, 0.08);
}}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================
def fmt_short(x: float | int | None) -> str:
    """Formatea números grandes a formato compacto: 1.2k / 3.4M / 5.6B."""
    if x is None:
        return "—"
    if isinstance(x, float) and np.isnan(x):
        return "—"
    x = float(x)
    ax = abs(x)
    if ax >= 1e9:
        return f"{x/1e9:.1f}B"
    if ax >= 1e6:
        return f"{x/1e6:.1f}M"
    if ax >= 1e3:
        return f"{x/1e3:.1f}k"
    return f"{x:.0f}"


def apply_plot_style(fig, x_title=None, y_title=None):
    """Aplica un estilo consistente para que todos los gráficos se vean iguales."""
    fig.update_layout(
        font=dict(family="Inter"),
        title=dict(font=dict(size=20)),
        margin=dict(t=70, r=25, b=45, l=55),
    )
    if x_title is not None:
        fig.update_xaxes(title=x_title)
    if y_title is not None:
        fig.update_yaxes(title=y_title)
    return fig


def _download(url: str, dest: Path) -> None:
    """Descarga un fichero a disco (simple y sin dependencias extra)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)  # noqa: S310


def _extract_zip(zip_path: Path, out_dir: Path) -> None:
    """Extrae un zip en out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def _find_first_csv(folder: Path) -> Path | None:
    """Devuelve el primer .csv que encuentre dentro de una carpeta."""
    csvs = list(folder.rglob("*.csv"))
    return csvs[0] if csvs else None


@st.cache_data(show_spinner=True)
def ensure_csvs() -> tuple[Path, Path]:
    """
    Asegura que existen 2 CSV disponibles para leer.
    Prioridad:
      1) Si hay parte_1.zip / parte_2.zip en el repo -> los extrae.
      2) Si no existen, los descarga desde ZIP_URL_1 / ZIP_URL_2 y los extrae.
    """
    DATA_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)

    # Si ya existen CSVs (por ejemplo porque tú los pusiste en /data), los usamos.
    csv1_existing = DATA_DIR / "parte_1.csv"
    csv2_existing = DATA_DIR / "parte_2.csv"
    if csv1_existing.exists() and csv2_existing.exists():
        return csv1_existing, csv2_existing

    # Carpeta donde dejamos los zips extraídos
    out1 = CACHE_DIR / "parte_1_extracted"
    out2 = CACHE_DIR / "parte_2_extracted"
    out1.mkdir(parents=True, exist_ok=True)
    out2.mkdir(parents=True, exist_ok=True)

    # 1) ZIP local en repo
    if LOCAL_ZIP_1.exists() and LOCAL_ZIP_2.exists():
        zip1 = LOCAL_ZIP_1
        zip2 = LOCAL_ZIP_2
    else:
        # 2) ZIP remoto (Releases)
        zip1 = CACHE_DIR / "parte_1.zip"
        zip2 = CACHE_DIR / "parte_2.zip"
        if not zip1.exists():
            _download(ZIP_URL_1, zip1)
        if not zip2.exists():
            _download(ZIP_URL_2, zip2)

    # Extraemos (si aún no hay CSV extraído)
    if _find_first_csv(out1) is None:
        _extract_zip(zip1, out1)
    if _find_first_csv(out2) is None:
        _extract_zip(zip2, out2)

    csv1 = _find_first_csv(out1)
    csv2 = _find_first_csv(out2)

    if csv1 is None or csv2 is None:
        raise RuntimeError("No se encontraron CSV dentro de los ZIPs. Revisa los archivos.")

    return csv1, csv2


@st.cache_data(show_spinner=True)
def load_data() -> pd.DataFrame:
    """Carga los CSV (desde zip) y prepara tipos/columnas derivadas optimizando memoria."""
    csv1, csv2 = ensure_csvs()

    # low_memory=False reduce los warnings de dtype mixto y suele ser más estable en Cloud
    df1 = pd.read_csv(csv1, low_memory=False)
    df2 = pd.read_csv(csv2, low_memory=False)

    # Si tienen columnas distintas, nos quedamos con las comunes
    if set(df1.columns) == set(df2.columns):
        df = pd.concat([df1, df2], ignore_index=True)
    else:
        common_cols = list(set(df1.columns).intersection(set(df2.columns)))
        if not common_cols:
            raise RuntimeError("Los dos CSV no tienen columnas comunes. Revisa el dataset.")
        df = pd.concat([df1[common_cols], df2[common_cols]], ignore_index=True)
        st.warning("⚠️ Los CSV no tenían las mismas columnas; he unido solo las columnas comunes.")

    # Tipos básicos
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "sales" in df.columns:
        df["sales"] = pd.to_numeric(df["sales"], errors="coerce").fillna(0).astype("float32")

    if "onpromotion" in df.columns:
        df["onpromotion"] = pd.to_numeric(df["onpromotion"], errors="coerce").fillna(0).astype("int16")

    if "transactions" in df.columns:
        df["transactions"] = pd.to_numeric(df["transactions"], errors="coerce").fillna(0).astype("float32")

    # Derivadas
    if "date" in df.columns and df["date"].notna().any():
        if "year" not in df.columns:
            df["year"] = df["date"].dt.year.astype("int16")
        if "month" not in df.columns:
            df["month"] = df["date"].dt.month.astype("int8")
        if "week" not in df.columns:
            df["week"] = df["date"].dt.isocalendar().week.astype("int16")
        if "day_of_week" not in df.columns:
            df["day_of_week"] = df["date"].dt.day_name()

    # Pequeña optimización: categorías (baja mucha memoria)
    for col in ["family", "state", "store_type", "holiday_type", "day_of_week"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    if "store_nbr" in df.columns:
        # Si viene como float, lo arreglamos
        df["store_nbr"] = pd.to_numeric(df["store_nbr"], errors="coerce").astype("Int16")

    return df


# ============================================================
# LOAD
# ============================================================
try:
    df = load_data()
except Exception as e:
    st.error(f"❌ Error cargando datos: {e}")
    st.stop()

# ============================================================
# SIDEBAR: filtros + navegación (en vez de tabs para que NO reviente Cloud)
# ============================================================
st.sidebar.markdown("<span class='badge'>Filtros</span>", unsafe_allow_html=True)

if "date" in df.columns and df["date"].notna().any():
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()

    date_range = st.sidebar.date_input(
        "Rango de fechas",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    df_f = df[(df["date"].dt.date >= date_range[0]) & (df["date"].dt.date <= date_range[1])].copy()
else:
    st.sidebar.warning("No hay columna date válida. Se usarán todos los datos.")
    df_f = df.copy()

if "year" in df_f.columns:
    years = sorted(pd.Series(df_f["year"]).dropna().unique().tolist())
    year_sel = st.sidebar.multiselect("Años", years, default=years)
    if year_sel:
        df_f = df_f[df_f["year"].isin(year_sel)].copy()

st.sidebar.markdown(
    "<div class='small-note'>Tip: usa filtros para que rankings y medias sean coherentes.</div>",
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navegación",
    ["1️⃣ Vista Global", "2️⃣ Por Tienda", "3️⃣ Por Estado", "4️⃣ Insights Extra"],
    index=0,
)

# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
<div class="banner">
  <div class="banner-title">📊 Dashboard de Ventas — Vista ejecutiva</div>
  <div class="banner-sub">Explora KPIs, estacionalidad, promociones y comparativas por tienda/estado.</div>
</div>
""",
    unsafe_allow_html=True,
)
st.caption("Producto = family · Ventas en promoción = registros con onpromotion > 0")


# ============================================================
# PAGE 1: VISTA GLOBAL
# ============================================================
def render_vista_global(data: pd.DataFrame) -> None:
    st.header("1️⃣ Vista Global")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Número total de tiendas", data["store_nbr"].nunique() if "store_nbr" in data.columns else "N/A")
    c2.metric("Número total de productos", data["family"].nunique() if "family" in data.columns else "N/A")
    c3.metric("Estados", data["state"].nunique() if "state" in data.columns else "N/A")
    c4.metric("Meses con datos", data["date"].dt.to_period("M").nunique() if "date" in data.columns else "N/A")

    st.markdown(
        "<div class='small-note'>📌 KPIs para entender tamaño del negocio y cobertura temporal del dataset.</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # Top 10 productos
    if "family" in data.columns and "sales" in data.columns:
        st.subheader("Top productos")
        st.markdown(
            "<div class='small-note'>Cómo leerlo: ranking global de familias por ventas (útil para priorizar surtido).</div>",
            unsafe_allow_html=True,
        )
        top_products = (
            data.groupby("family", as_index=False)["sales"].sum()
            .sort_values("sales", ascending=False)
            .head(10)
        )
        fig = px.bar(
            top_products,
            x="sales",
            y="family",
            orientation="h",
            title="Top 10 productos más vendidos (ventas)",
            color_discrete_sequence=[ACCENT_TURQ],
        )
        fig.update_xaxes(tickformat=".2s")
        apply_plot_style(fig, x_title="Ventas", y_title="Producto (family)")
        st.plotly_chart(fig, width="stretch")

    # Distribución por tienda
    if "store_nbr" in data.columns and "sales" in data.columns:
        st.subheader("Distribución de ventas por tienda")
        st.markdown(
            "<div class='small-note'>ECDF: qué % de tiendas está por debajo de X ventas; Boxplot: resumen + outliers.</div>",
            unsafe_allow_html=True,
        )

        sales_by_store = data.groupby("store_nbr", as_index=False)["sales"].sum()
        colA, colB = st.columns(2)

        with colA:
            fig = px.ecdf(
                sales_by_store,
                x="sales",
                title="Distribución (ECDF)",
                markers=False,
                color_discrete_sequence=[ACCENT_TURQ],
            )
            fig.update_xaxes(tickformat=".2s")
            apply_plot_style(fig, x_title="Ventas totales por tienda", y_title="% acumulado de tiendas")
            st.plotly_chart(fig, width="stretch")

        with colB:
            fig = px.box(
                sales_by_store,
                y="sales",
                points="all",
                title="Resumen (Boxplot + puntos)",
                color_discrete_sequence=[ACCENT_ORANGE],
            )
            fig.update_yaxes(tickformat=".2s")
            apply_plot_style(fig, y_title="Ventas totales por tienda")
            st.plotly_chart(fig, width="stretch")

    # Estacionalidad semanal y mensual (light)
    if "day_of_week" in data.columns and "sales" in data.columns:
        st.subheader("Estacionalidad semanal")
        order_en = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow = data.groupby("day_of_week", as_index=False)["sales"].mean()

        fig = px.bar(
            dow,
            x="day_of_week",
            y="sales",
            title="Ventas medias por día de la semana",
            category_orders={"day_of_week": order_en},
            color_discrete_sequence=[ACCENT_TURQ],
            text="sales",
        )
        fig.update_traces(texttemplate="%{text:.0f}", textposition="outside", cliponaxis=False)
        apply_plot_style(fig, x_title="Día", y_title="Ventas medias")
        st.plotly_chart(fig, width="stretch")


# ============================================================
# PAGE 2: POR TIENDA
# ============================================================
def render_por_tienda(data: pd.DataFrame) -> None:
    st.header("2️⃣ Por Tienda (store_nbr)")
    st.markdown(
        "<div class='small-note'>📌 Entender el rendimiento de una tienda concreta.</div>",
        unsafe_allow_html=True,
    )

    if "store_nbr" not in data.columns:
        st.error("Falta la columna store_nbr.")
        return

    stores = sorted(pd.Series(data["store_nbr"]).dropna().unique().tolist())
    store_sel = st.selectbox("Selecciona una tienda", stores)

    sdf = data[data["store_nbr"] == store_sel].copy()

    if "year" in sdf.columns and "sales" in sdf.columns:
        sales_year = sdf.groupby("year", as_index=False)["sales"].sum().sort_values("year")
        fig = px.bar(
            sales_year,
            x="year",
            y="sales",
            title=f"Ventas totales por año — Tienda {store_sel}",
            color_discrete_sequence=[ACCENT_TURQ],
            text="sales",
        )
        fig.update_traces(texttemplate="%{text:.2s}", textposition="outside", cliponaxis=False)
        fig.update_yaxes(tickformat=".2s")
        apply_plot_style(fig, x_title="Año", y_title="Ventas")
        st.plotly_chart(fig, width="stretch")

    prod_count = sdf.loc[sdf["sales"] > 0, "family"].nunique() if ("family" in sdf.columns and "sales" in sdf.columns) else None
    prod_promo = (
        sdf.loc[(sdf["sales"] > 0) & (sdf["onpromotion"] > 0), "family"].nunique()
        if ("family" in sdf.columns and "sales" in sdf.columns and "onpromotion" in sdf.columns)
        else None
    )

    c1, c2 = st.columns(2)
    c1.metric("Número total de productos vendidos", "N/A" if prod_count is None else prod_count)
    c2.metric("Productos vendidos en promoción", "N/A" if prod_promo is None else prod_promo)


# ============================================================
# PAGE 3: POR ESTADO
# ============================================================
def render_por_estado(data: pd.DataFrame) -> None:
    st.header("3️⃣ Por Estado (state)")
    st.markdown(
        "<div class='small-note'>📌 Rendimiento regional: transacciones, top tiendas y top familias.</div>",
        unsafe_allow_html=True,
    )

    if "state" not in data.columns:
        st.error("Falta la columna state.")
        return

    states = sorted(pd.Series(data["state"]).dropna().unique().tolist())
    state_sel = st.selectbox("Selecciona un estado", states)

    edf = data[data["state"] == state_sel].copy()

    if "transactions" in edf.columns and "year" in edf.columns:
        tx_year = edf.groupby("year", as_index=False)["transactions"].sum().sort_values("year")
        tx_year["tx_label"] = tx_year["transactions"].apply(lambda v: f"{v/1e6:.1f}M")

        fig = px.line(
            tx_year,
            x="year",
            y="transactions",
            markers=True,
            text="tx_label",
            title=f"Transacciones por año — {state_sel}",
            color_discrete_sequence=[ACCENT_TURQ],
        )
        fig.update_traces(mode="lines+markers+text", textposition="top center", cliponaxis=False)
        fig.update_yaxes(tickformat=".2s")
        apply_plot_style(fig, x_title="Año", y_title="Transacciones")
        st.plotly_chart(fig, width="stretch")

    if "store_nbr" in edf.columns and "sales" in edf.columns:
        top_state_stores = (
            edf.groupby("store_nbr", as_index=False)["sales"].sum()
            .sort_values("sales", ascending=False)
            .head(10)
        )
        fig = px.bar(
            top_state_stores,
            x="store_nbr",
            y="sales",
            title=f"Top 10 tiendas con más ventas — {state_sel}",
            color_discrete_sequence=[ACCENT_ORANGE],
        )
        fig.update_yaxes(tickformat=".2s")
        apply_plot_style(fig, x_title="Tienda", y_title="Ventas")
        st.plotly_chart(fig, width="stretch")


# ============================================================
# PAGE 4: INSIGHTS EXTRA (animación + comparador)
# ============================================================
def render_insights(data: pd.DataFrame) -> None:
    st.header("4️⃣ Insights Extra (para sorprender)")

    total_sales = data["sales"].sum() if "sales" in data.columns else None
    total_tx = data["transactions"].sum() if "transactions" in data.columns else None
    promo_share = (data["onpromotion"].gt(0).mean() * 100) if "onpromotion" in data.columns else None

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Ventas totales", fmt_short(total_sales))
    k2.metric("Transacciones", fmt_short(total_tx))
    k3.metric("% registros con promo", "—" if promo_share is None else f"{promo_share:.1f}%")
    k4.metric("Venta media por registro", fmt_short(data["sales"].mean()) if "sales" in data.columns else "—")

    st.divider()

    # Animación: la dejamos opcional porque puede tumbar Cloud si está a tope
    st.subheader("🎞️ Animación: Top productos por año")
    st.markdown(
        "<div class='small-note'>Tip: actívala solo si ves que la app va fluida (consume recursos).</div>",
        unsafe_allow_html=True,
    )

    do_anim = st.toggle("Activar animación (consume recursos)", value=False)

    if do_anim and all(col in data.columns for col in ["year", "family", "sales"]):
        yfs = (
            data.groupby(["year", "family"], as_index=False)["sales"].sum()
            .sort_values(["year", "sales"], ascending=[True, False])
        )
        yfs["rank"] = yfs.groupby("year")["sales"].rank(method="first", ascending=False)
        yfs_top = yfs[yfs["rank"] <= 10].copy()

        fig = px.bar(
            yfs_top,
            x="sales",
            y="family",
            orientation="h",
            animation_frame="year",
            animation_group="family",
            title="Top 10 productos (family) por ventas — Animado por año",
            color_discrete_sequence=[ACCENT_TURQ],
        )
        fig.update_xaxes(tickformat=".2s")
        apply_plot_style(fig, x_title="Ventas", y_title="Producto")
        st.plotly_chart(fig, width="stretch")
    elif do_anim:
        st.info("No hay columnas suficientes para animación (necesito year, family y sales).")

    st.divider()

    # Comparador Tienda A vs Tienda B
    st.subheader("🆚 Comparador: Tienda A vs Tienda B")
    st.markdown(
        "<div class='small-note'>Compara KPIs y evolución mensual entre dos tiendas.</div>",
        unsafe_allow_html=True,
    )

    if "store_nbr" not in data.columns:
        st.info("No hay columna store_nbr para comparador.")
        return

    stores = sorted(pd.Series(data["store_nbr"]).dropna().unique().tolist())
    col1, col2 = st.columns(2)
    with col1:
        store_a = st.selectbox("Tienda A", stores, index=0)
    with col2:
        store_b = st.selectbox("Tienda B", stores, index=1 if len(stores) > 1 else 0)

    a_df = data[data["store_nbr"] == store_a].copy()
    b_df = data[data["store_nbr"] == store_b].copy()

    a_sales = a_df["sales"].sum() if "sales" in a_df.columns else np.nan
    b_sales = b_df["sales"].sum() if "sales" in b_df.columns else np.nan

    kA, kB = st.columns(2)
    kA.metric("Ventas Tienda A", fmt_short(a_sales), delta=fmt_short(a_sales - b_sales) if not np.isnan(b_sales) else None)
    kB.metric("Ventas Tienda B", fmt_short(b_sales), delta=fmt_short(b_sales - a_sales) if not np.isnan(a_sales) else None)

    if "date" in data.columns and "sales" in data.columns:
        a_m = a_df.dropna(subset=["date"]).copy()
        b_m = b_df.dropna(subset=["date"]).copy()

        a_m["month_period"] = a_m["date"].dt.to_period("M").astype(str)
        b_m["month_period"] = b_m["date"].dt.to_period("M").astype(str)

        a_series = a_m.groupby("month_period", as_index=False)["sales"].sum()
        b_series = b_m.groupby("month_period", as_index=False)["sales"].sum()

        series = a_series.merge(b_series, on="month_period", how="outer", suffixes=("_A", "_B")).fillna(0)
        series = series.sort_values("month_period")

        series_long = series.melt("month_period", value_vars=["sales_A", "sales_B"], var_name="store", value_name="sales")
        series_long["store"] = series_long["store"].map({"sales_A": f"Tienda {store_a}", "sales_B": f"Tienda {store_b}"})

        fig = px.line(
            series_long,
            x="month_period",
            y="sales",
            color="store",
            markers=True,
            title="Comparación de ventas (mensual) — Tienda A vs Tienda B",
            color_discrete_sequence=[ACCENT_TURQ, ACCENT_ORANGE],
        )
        fig.update_yaxes(tickformat=".2s")
        apply_plot_style(fig, x_title="Mes", y_title="Ventas")
        st.plotly_chart(fig, width="stretch")


# ============================================================
# ROUTER (solo renderiza la página seleccionada -> clave para que NO desaparezca)
# ============================================================
if page == "1️⃣ Vista Global":
    render_vista_global(df_f)
elif page == "2️⃣ Por Tienda":
    render_por_tienda(df_f)
elif page == "3️⃣ Por Estado":
    render_por_estado(df_f)
else:
    render_insights(df_f)
