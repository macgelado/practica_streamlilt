"""
Dashboard de Ventas (Práctica Streamlit)

Este script construye un dashboard interactivo en Streamlit para analizar ventas.
Incluye 4 pestañas (según el enunciado) y añade extras "executive-ready":
- Estilo visual (turquesa + naranja suave) y tipografía.
- Gráficos con contexto (micro-narrativa) para que se entiendan rápido.
- Animación con Plotly (top productos por año).
- Comparador (Store A vs Store B) en la pestaña 4.

Cómo ejecutar:
    python -m streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ============================================================
# THEME (turquesa + naranja suave) + tipografía
# ============================================================
ACCENT_TURQ = "#1FB6AA"
ACCENT_ORANGE = "#FFB168"
ACCENT_DARK = "#0B1320"
ACCENT_MID = "#1A2639"

ZIP_URL_1 = "https://github.com/macgelado/practica_streamlilt/releases/download/v1/parte_1.zip"
ZIP_URL_2 = "https://github.com/macgelado/practica_streamlilt/releases/download/v1/parte_2.zip"

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

# Estilo general + "idea 10": look más limpio tipo web (ocultar menú/footer, etc.)
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

/* Ocultar elementos "marca Streamlit" para que parezca más web-app (idea 10) */
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

h1, h2, h3 {{
  letter-spacing: -0.02em;
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

/* Tabs un poco más "card" */
div[data-testid="stTabs"] button {{
  border-radius: 12px !important;
  padding: 10px 14px !important;
}}
</style>
""",
    unsafe_allow_html=True,
)


def fmt_short(x: float) -> str:
    """Formatea números grandes a formato compacto: 1.2k / 3.4M / 5.6B."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
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
    """Aplica un estilo consistente para que todos los gráficos respiren igual."""
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


# ============================================================
# LOAD DATA
# ============================================================
from pathlib import Path
from io import BytesIO
import zipfile
import requests

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

@st.cache_data(show_spinner=True)
def _download_and_extract_csv(zip_url: str, out_csv_name: str) -> Path:
    """
    Descarga un ZIP desde GitHub Releases y extrae el primer CSV que encuentre.
    Devuelve la ruta local del CSV extraído.
    """
    out_csv = DATA_DIR / out_csv_name
    if out_csv.exists():
        return out_csv

    # Descarga robusta (GitHub a veces redirige)
    r = requests.get(zip_url, allow_redirects=True, timeout=180)
    r.raise_for_status()

    with zipfile.ZipFile(BytesIO(r.content)) as z:
        csv_members = [m for m in z.namelist() if m.lower().endswith(".csv")]
        if not csv_members:
            raise ValueError(f"No hay ningún CSV dentro del ZIP: {zip_url}")
        member = csv_members[0]

        with z.open(member) as f_in, open(out_csv, "wb") as f_out:
            f_out.write(f_in.read())

    return out_csv


@st.cache_data(show_spinner=True)
def load_data():
    """
    Carga datos desde los ZIP de Releases (v1), prepara tipos y columnas derivadas.
    """
    # Mensajes de “debug” para Streamlit Cloud (se ven en la app)
    st.info("📥 Descargando y extrayendo datos desde GitHub Releases...")

    csv1 = _download_and_extract_csv(ZIP_URL_1, "parte_1.csv")
    st.success(f"✅ Extraído: {csv1}")

    csv2 = _download_and_extract_csv(ZIP_URL_2, "parte_2.csv")
    st.success(f"✅ Extraído: {csv2}")

    st.info("📄 Leyendo CSVs (esto puede tardar un poco)...")

    df1 = pd.read_csv(csv1, low_memory=False)
    df2 = pd.read_csv(csv2, low_memory=False)

    # Si tienen mismas columnas -> concat
    if set(df1.columns) == set(df2.columns):
        df = pd.concat([df1, df2], ignore_index=True)
    else:
        common_cols = list(set(df1.columns).intersection(set(df2.columns)))
        if common_cols:
            df = pd.concat([df1[common_cols], df2[common_cols]], ignore_index=True)
        else:
            raise ValueError("No hay columnas comunes entre parte_1 y parte_2.")

    # Tipos básicos
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "sales" in df.columns:
        df["sales"] = pd.to_numeric(df["sales"], errors="coerce").fillna(0)

    if "onpromotion" in df.columns:
        df["onpromotion"] = pd.to_numeric(df["onpromotion"], errors="coerce").fillna(0).astype(int)

    if "transactions" in df.columns:
        df["transactions"] = pd.to_numeric(df["transactions"], errors="coerce").fillna(0)

    # Derivadas
    if "date" in df.columns and df["date"].notna().any():
        if "year" not in df.columns:
            df["year"] = df["date"].dt.year
        if "month" not in df.columns:
            df["month"] = df["date"].dt.month
        if "week" not in df.columns:
            df["week"] = df["date"].dt.isocalendar().week.astype(int)
        if "day_of_week" not in df.columns:
            df["day_of_week"] = df["date"].dt.day_name()

    st.success("✅ Datos cargados correctamente.")
    return df



df = load_data()
if df is None:
    st.stop()

# ============================================================
# SIDEBAR FILTERS
# ============================================================
st.sidebar.markdown("<span class='badge'>Filtros</span>", unsafe_allow_html=True)
st.sidebar.write("")

if "date" in df.columns and df["date"].notna().any():
    min_date = df["date"].min()
    max_date = df["date"].max()

    date_range = st.sidebar.date_input(
        "Rango de fechas",
        value=(min_date.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date(),
    )

    df_f = df[(df["date"].dt.date >= date_range[0]) & (df["date"].dt.date <= date_range[1])].copy()
else:
    st.sidebar.warning("No hay columna date válida. Se usarán todos los datos.")
    df_f = df.copy()

if "year" in df_f.columns:
    years = sorted(df_f["year"].dropna().unique().tolist())
    year_sel = st.sidebar.multiselect("Años", years, default=years)
    df_f = df_f[df_f["year"].isin(year_sel)].copy()

st.sidebar.write("")
st.sidebar.markdown(
    "<div class='small-note'>Tip: usa los filtros para que rankings y medias sean coherentes.</div>",
    unsafe_allow_html=True,
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
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "1️⃣ Vista Global",
        "2️⃣ Por Tienda",
        "3️⃣ Por Estado",
        "4️⃣ Insights Extra",
    ]
)

# ============================================================
# TAB 1
# ============================================================
with tab1:
    st.header("1️⃣ Vista Global")

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Número total de tiendas", df_f["store_nbr"].nunique() if "store_nbr" in df_f.columns else "N/A")
    c2.metric("Número total de productos", df_f["family"].nunique() if "family" in df_f.columns else "N/A")
    c3.metric("Estados", df_f["state"].nunique() if "state" in df_f.columns else "N/A")
    c4.metric("Meses con datos", df_f["date"].dt.to_period("M").nunique() if "date" in df_f.columns else "N/A")

    st.markdown(
        "<div class='small-note'>📌 Estos KPIs te dicen el tamaño del negocio y el alcance temporal del dataset.</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # Top 10 productos (global)
    if "family" in df_f.columns and "sales" in df_f.columns:
        st.subheader("Top productos")
        st.markdown(
            "<div class='small-note'>Cómo leerlo: ranking global de familias por ventas (útil para priorizar surtido).</div>",
            unsafe_allow_html=True,
        )

        top_products = (
            df_f.groupby("family", as_index=False)["sales"].sum()
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
        st.plotly_chart(fig, use_container_width=True)

    # Distribución por tienda (ECDF + Box)
    if "store_nbr" in df_f.columns and "sales" in df_f.columns:
        st.subheader("Distribución de ventas por tienda")
        st.markdown(
            "<div class='small-note'>Cómo leerlo: ECDF muestra qué % de tiendas está por debajo de X ventas; el boxplot resume outliers.</div>",
            unsafe_allow_html=True,
        )

        # Agrupamos por tienda para comparar “tiendas” (no registros).
        sales_by_store = df_f.groupby("store_nbr", as_index=False)["sales"].sum()
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
            st.plotly_chart(fig, use_container_width=True)

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
            st.plotly_chart(fig, use_container_width=True)

    # Top 10 tiendas promo
    if "store_nbr" in df_f.columns and "sales" in df_f.columns and "onpromotion" in df_f.columns:
        st.subheader("Promociones")
        st.markdown(
            "<div class='small-note'>Cómo leerlo: tiendas donde las promociones generan más ventas totales (potenciales 'best practices').</div>",
            unsafe_allow_html=True,
        )

        promo = df_f[df_f["onpromotion"] > 0]
        top_stores_promo = (
            promo.groupby("store_nbr", as_index=False)["sales"].sum()
            .sort_values("sales", ascending=False)
            .head(10)
        )
        fig = px.bar(
            top_stores_promo,
            x="store_nbr",
            y="sales",
            title="Top 10 tiendas con más ventas en promoción",
            color_discrete_sequence=[ACCENT_ORANGE],
        )
        fig.update_yaxes(tickformat=".2s")
        apply_plot_style(fig, x_title="Tienda", y_title="Ventas en promo")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Día de la semana con etiquetas encima
    if "day_of_week" in df_f.columns and "sales" in df_f.columns:
        st.subheader("Estacionalidad semanal")
        st.markdown(
            "<div class='small-note'>Cómo leerlo: ventas medias por día (útil para planificar personal/stock).</div>",
            unsafe_allow_html=True,
        )

        order_en = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow = df_f.groupby("day_of_week", as_index=False)["sales"].mean()

        fig = px.bar(
            dow,
            x="day_of_week",
            y="sales",
            title="Día de la semana con más ventas (término medio)",
            category_orders={"day_of_week": order_en},
            text="sales",
            color_discrete_sequence=[ACCENT_TURQ],
        )
        fig.update_traces(texttemplate="%{text:.0f}", textposition="outside", cliponaxis=False)
        apply_plot_style(fig, x_title="Día", y_title="Ventas medias")
        st.plotly_chart(fig, use_container_width=True)

    # Ventas medias por semana
    if "year" in df_f.columns and "week" in df_f.columns and "sales" in df_f.columns:
        st.subheader("Estacionalidad por semana")
        st.markdown(
            "<div class='small-note'>Cómo leerlo: compara patrones por año y detecta picos recurrentes.</div>",
            unsafe_allow_html=True,
        )

        weekly = df_f.groupby(["year", "week"], as_index=False)["sales"].mean()
        fig = px.line(
            weekly,
            x="week",
            y="sales",
            color="year",
            title="Ventas medias por semana (comparativa por año)",
        )
        fig.update_yaxes(tickformat=".2s")
        apply_plot_style(fig, x_title="Semana", y_title="Ventas medias")
        st.plotly_chart(fig, use_container_width=True)

    # Ventas medias por mes
    if "year" in df_f.columns and "month" in df_f.columns and "sales" in df_f.columns:
        st.subheader("Estacionalidad por mes")
        st.markdown(
            "<div class='small-note'>Cómo leerlo: visión mensual para campañas y previsión (se ve rápido la estacionalidad).</div>",
            unsafe_allow_html=True,
        )

        monthly = df_f.groupby(["year", "month"], as_index=False)["sales"].mean()
        fig = px.line(
            monthly,
            x="month",
            y="sales",
            color="year",
            markers=True,
            title="Ventas medias por mes (comparativa por año)",
        )
        fig.update_yaxes(tickformat=".2s")
        apply_plot_style(fig, x_title="Mes", y_title="Ventas medias")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 2
# ============================================================
with tab2:
    st.header("2️⃣ Por Tienda (store_nbr)")
    st.markdown(
        "<div class='small-note'>📌 Esta pestaña sirve para entender el rendimiento de una tienda concreta.</div>",
        unsafe_allow_html=True,
    )

    if "store_nbr" not in df_f.columns:
        st.error("Falta la columna store_nbr en el dataset.")
    else:
        stores = sorted(df_f["store_nbr"].dropna().unique().tolist())
        store_sel = st.selectbox("Selecciona una tienda", stores)

        sdf = df_f[df_f["store_nbr"] == store_sel].copy()

        # a) ventas por año
        if "year" in sdf.columns and "sales" in sdf.columns:
            sales_year = sdf.groupby("year", as_index=False)["sales"].sum().sort_values("year")
            sales_year["year"] = sales_year["year"].astype(int)

            fig = px.bar(
                sales_year,
                x="year",
                y="sales",
                title=f"Ventas totales por año - Tienda {store_sel}",
                color_discrete_sequence=[ACCENT_TURQ],
                text="sales",
            )
            fig.update_traces(texttemplate="%{text:.2s}", textposition="outside", cliponaxis=False)
            fig.update_xaxes(tickmode="linear", dtick=1)
            fig.update_yaxes(tickformat=".2s")
            apply_plot_style(fig, x_title="Año", y_title="Ventas")
            st.plotly_chart(fig, use_container_width=True)

        # b) productos vendidos (familias con ventas)
        prod_count = (
            sdf.loc[sdf["sales"] > 0, "family"].nunique()
            if ("family" in sdf.columns and "sales" in sdf.columns)
            else None
        )

        # c) productos vendidos en promo
        prod_promo = (
            sdf.loc[(sdf["sales"] > 0) & (sdf["onpromotion"] > 0), "family"].nunique()
            if ("family" in sdf.columns and "sales" in sdf.columns and "onpromotion" in sdf.columns)
            else None
        )

        c1, c2 = st.columns(2)
        c1.metric("Número total de productos vendidos", "N/A" if prod_count is None else prod_count)
        c2.metric("Productos vendidos en promoción", "N/A" if prod_promo is None else prod_promo)

# ============================================================
# TAB 3
# ============================================================
with tab3:
    st.header("3️⃣ Por Estado (state)")
    st.markdown(
        "<div class='small-note'>📌 Esta pestaña permite revisar rendimiento regional (transacciones, top tiendas y top familias).</div>",
        unsafe_allow_html=True,
    )

    if "state" not in df_f.columns:
        st.error("Falta la columna state en el dataset.")
    else:
        states = sorted(df_f["state"].dropna().unique().tolist())
        state_sel = st.selectbox("Selecciona un estado", states)

        edf = df_f[df_f["state"] == state_sel].copy()

        # a) transacciones por año (etiquetas + eje X entero)
        if "transactions" in edf.columns and "year" in edf.columns:
            tx_year = edf.groupby("year", as_index=False)["transactions"].sum().sort_values("year")
            tx_year["year"] = tx_year["year"].astype(int)
            tx_year["tx_label"] = tx_year["transactions"].apply(lambda v: f"{v/1e6:.1f}M")

            st.markdown(
                "<div class='small-note'>Cómo leerlo: volumen de actividad (transacciones) por año para el estado seleccionado.</div>",
                unsafe_allow_html=True,
            )

            fig = px.line(
                tx_year,
                x="year",
                y="transactions",
                markers=True,
                text="tx_label",
                title=f"Transacciones por año - {state_sel}",
                color_discrete_sequence=[ACCENT_TURQ],
            )
            fig.update_traces(mode="lines+markers+text", textposition="top center", cliponaxis=False)
            fig.update_xaxes(tickmode="linear", dtick=1)
            fig.update_yaxes(tickformat=".2s")
            apply_plot_style(fig, x_title="Año", y_title="Transacciones")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No encuentro 'transactions' o 'year' para el gráfico de transacciones por año.")

        # b) ranking tiendas con más ventas
        if "store_nbr" in edf.columns and "sales" in edf.columns:
            st.markdown(
                "<div class='small-note'>Cómo leerlo: ranking de tiendas dentro del estado. Útil para detectar 'top performers'.</div>",
                unsafe_allow_html=True,
            )

            top_state_stores = (
                edf.groupby("store_nbr", as_index=False)["sales"].sum()
                .sort_values("sales", ascending=False)
                .head(10)
            )
            fig = px.bar(
                top_state_stores,
                x="store_nbr",
                y="sales",
                title=f"Top 10 tiendas con más ventas - {state_sel}",
                color_discrete_sequence=[ACCENT_ORANGE],
            )
            fig.update_yaxes(tickformat=".2s")
            apply_plot_style(fig, x_title="Tienda", y_title="Ventas")
            st.plotly_chart(fig, use_container_width=True)

        # c) producto más vendido + top 5
        if "family" in edf.columns and "sales" in edf.columns:
            fam_state = edf.groupby("family", as_index=False)["sales"].sum().sort_values("sales", ascending=False)

            if len(fam_state):
                top1 = fam_state.iloc[0]
                st.success(
                    f"Producto más vendido en {state_sel}: **{top1['family']}** (ventas: {top1['sales']:.2f})"
                )

                st.markdown(
                    "<div class='small-note'>Nota: si ves que siempre gana <b>GROCERY I</b>, suele ser normal porque agrupa muchos básicos.</div>",
                    unsafe_allow_html=True,
                )

                top5 = fam_state.head(5).copy()
                fig = px.bar(
                    top5.sort_values("sales"),
                    x="sales",
                    y="family",
                    orientation="h",
                    title=f"Top 5 productos (family) en {state_sel}",
                    color_discrete_sequence=[ACCENT_TURQ],
                    text="sales",
                )
                fig.update_traces(texttemplate="%{text:.2s}", textposition="outside", cliponaxis=False)
                fig.update_xaxes(tickformat=".2s")
                apply_plot_style(fig, x_title="Ventas", y_title="Producto (family)")
                st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 4 (innovación: animación + comparador + narrativa)
# ============================================================
with tab4:
    st.header("4️⃣ Insights Extra (para sorprender)")
    st.markdown(
        "<div class='small-note'>Aquí añadimos funcionalidades más 'dashboard': animaciones, comparadores y lectura ejecutiva.</div>",
        unsafe_allow_html=True,
    )

    # KPIs ejecutivos
    total_sales = df_f["sales"].sum() if "sales" in df_f.columns else None
    total_tx = df_f["transactions"].sum() if "transactions" in df_f.columns else None
    promo_share = (df_f["onpromotion"].gt(0).mean() * 100) if "onpromotion" in df_f.columns else None

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Ventas totales", fmt_short(total_sales))
    k2.metric("Transacciones", fmt_short(total_tx))
    k3.metric("% registros con promo", "—" if promo_share is None else f"{promo_share:.1f}%")
    k4.metric("Venta media por registro", fmt_short(df_f["sales"].mean()) if "sales" in df_f.columns else "—")

    st.divider()

    # ------------------------------------------------------------
    # IDEA 3) ANIMACIÓN: Top productos por año (Plotly animation)
    # ------------------------------------------------------------
    st.subheader("🎞️ Animación: Top productos por año")
    st.markdown(
        "<div class='small-note'>Cómo leerlo: cada año (frame) muestra el ranking de familias por ventas. Es muy visual para ver cambios en el mix.</div>",
        unsafe_allow_html=True,
    )

    if all(col in df_f.columns for col in ["year", "family", "sales"]):
        # Agrupamos por año+familia y rankeamos dentro de cada año (truco para quedarnos con Top 10 por frame)
        yfs = (
            df_f.groupby(["year", "family"], as_index=False)["sales"]
            .sum()
            .sort_values(["year", "sales"], ascending=[True, False])
        )
        yfs["rank"] = yfs.groupby("year")["sales"].rank(method="first", ascending=False)
        yfs_top = yfs[yfs["rank"] <= 10].copy()

        # Para que el eje Y no sea caótico, limitamos a las familias que entran en algún top10
        fam_pool = yfs_top["family"].unique().tolist()

        fig = px.bar(
            yfs_top,
            x="sales",
            y="family",
            orientation="h",
            animation_frame="year",
            animation_group="family",
            category_orders={"family": fam_pool},
            title="Top 10 productos (family) por ventas — Animado por año",
            color_discrete_sequence=[ACCENT_TURQ],
        )
        fig.update_xaxes(tickformat=".2s")
        # Acelerar un pelín la animación (queda más “pro”)
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="▶ Play",
                            method="animate",
                            args=[None, {"frame": {"duration": 700, "redraw": True}, "transition": {"duration": 200}}],
                        ),
                        dict(
                            label="⏸ Pause",
                            method="animate",
                            args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                        ),
                    ],
                )
            ]
        )
        apply_plot_style(fig, x_title="Ventas", y_title="Producto (family)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay columnas suficientes para animación (necesito year, family y sales).")

    st.divider()

    # ------------------------------------------------------------
    # IDEA 9) Contexto: micro-narrativa + "qué mirar"
    # ------------------------------------------------------------
    st.subheader("📌 Lectura rápida (contexto)")
    insights = []
    if "sales" in df_f.columns:
        # Pequeños insights “automáticos” (queda muy dashboard)
        if "state" in df_f.columns:
            state_share = (
                df_f.groupby("state", as_index=False)["sales"].sum().sort_values("sales", ascending=False)
            )
            if len(state_share):
                top_state = state_share.iloc[0]["state"]
                share = (state_share.iloc[0]["sales"] / state_share["sales"].sum()) * 100
                insights.append(f"🔥 El estado **{top_state}** concentra aprox. **{share:.1f}%** de las ventas en el rango filtrado.")

        if "onpromotion" in df_f.columns:
            promo_sales = df_f[df_f["onpromotion"] > 0]["sales"].sum()
            total = df_f["sales"].sum()
            if total > 0:
                insights.append(f"🏷️ Las ventas en promoción representan aprox. **{(promo_sales/total)*100:.1f}%** del total.")

    if insights:
        for it in insights:
            st.write(it)
    else:
        st.write("Ajusta filtros para generar insights automáticos.")

    st.divider()

    # ------------------------------------------------------------
    # IDEA 7) COMPARADOR (Store A vs Store B) EN TAB 4
    # ------------------------------------------------------------
    st.subheader("🆚 Comparador: Tienda A vs Tienda B")
    st.markdown(
        "<div class='small-note'>Cómo leerlo: compara evolución temporal, mix de productos y KPIs entre dos tiendas.</div>",
        unsafe_allow_html=True,
    )

    if "store_nbr" in df_f.columns:
        stores = sorted(df_f["store_nbr"].dropna().unique().tolist())
        comp_col1, comp_col2 = st.columns(2)
        with comp_col1:
            store_a = st.selectbox("Tienda A", stores, index=0)
        with comp_col2:
            # Intento elegir otra por defecto (si existe)
            default_idx = 1 if len(stores) > 1 else 0
            store_b = st.selectbox("Tienda B", stores, index=default_idx)

        a_df = df_f[df_f["store_nbr"] == store_a].copy()
        b_df = df_f[df_f["store_nbr"] == store_b].copy()

        # KPIs comparativos
        kA, kB, kC, kD = st.columns(4)

        a_sales = a_df["sales"].sum() if "sales" in a_df.columns else np.nan
        b_sales = b_df["sales"].sum() if "sales" in b_df.columns else np.nan
        a_promo = a_df["onpromotion"].gt(0).mean() * 100 if "onpromotion" in a_df.columns else np.nan
        b_promo = b_df["onpromotion"].gt(0).mean() * 100 if "onpromotion" in b_df.columns else np.nan

        # Un delta sencillo hace que parezca más BI
        kA.metric("Ventas Tienda A", fmt_short(a_sales), delta=None if np.isnan(a_sales) or np.isnan(b_sales) else fmt_short(a_sales - b_sales))
        kB.metric("Ventas Tienda B", fmt_short(b_sales), delta=None if np.isnan(a_sales) or np.isnan(b_sales) else fmt_short(b_sales - a_sales))
        kC.metric("% promo A", "—" if np.isnan(a_promo) else f"{a_promo:.1f}%")
        kD.metric("% promo B", "—" if np.isnan(b_promo) else f"{b_promo:.1f}%")

        # Evolución temporal comparada (mensual)
        if "date" in df_f.columns and "sales" in df_f.columns:
            # Nota: agrupar por mes para que sea legible (en daily es demasiado ruido)
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
            st.plotly_chart(fig, use_container_width=True)

        # Mix de productos (Top 5 por tienda)
        if "family" in df_f.columns and "sales" in df_f.columns:
            mix_col1, mix_col2 = st.columns(2)

            with mix_col1:
                top5_a = (
                    a_df.groupby("family", as_index=False)["sales"].sum()
                    .sort_values("sales", ascending=False)
                    .head(5)
                )
                fig = px.bar(
                    top5_a.sort_values("sales"),
                    x="sales",
                    y="family",
                    orientation="h",
                    title=f"Top 5 productos — Tienda {store_a}",
                    color_discrete_sequence=[ACCENT_TURQ],
                    text="sales",
                )
                fig.update_traces(texttemplate="%{text:.2s}", textposition="outside", cliponaxis=False)
                fig.update_xaxes(tickformat=".2s")
                apply_plot_style(fig, x_title="Ventas", y_title="Producto")
                st.plotly_chart(fig, use_container_width=True)

            with mix_col2:
                top5_b = (
                    b_df.groupby("family", as_index=False)["sales"].sum()
                    .sort_values("sales", ascending=False)
                    .head(5)
                )
                fig = px.bar(
                    top5_b.sort_values("sales"),
                    x="sales",
                    y="family",
                    orientation="h",
                    title=f"Top 5 productos — Tienda {store_b}",
                    color_discrete_sequence=[ACCENT_ORANGE],
                    text="sales",
                )
                fig.update_traces(texttemplate="%{text:.2s}", textposition="outside", cliponaxis=False)
                fig.update_xaxes(tickformat=".2s")
                apply_plot_style(fig, x_title="Ventas", y_title="Producto")
                st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No hay columna store_nbr para construir el comparador.")

    st.divider()

    # ------------------------------------------------------------
    # Extras visuales que ya tenías (pero con algo más de contexto)
    # ------------------------------------------------------------
    st.subheader("Otros insights rápidos")

    # Impacto promoción
    if "sales" in df_f.columns and "onpromotion" in df_f.columns:
        st.markdown(
            "<div class='small-note'>Cómo leerlo: compara ventas medias cuando hay promo vs cuando no (indicador de efectividad).</div>",
            unsafe_allow_html=True,
        )

        base = df_f.copy()
        base["promo_flag"] = np.where(base["onpromotion"] > 0, "Con promo", "Sin promo")
        promo_cmp = base.groupby("promo_flag", as_index=False)["sales"].mean()

        fig = px.bar(
            promo_cmp,
            x="promo_flag",
            y="sales",
            title="Impacto promoción: ventas medias (Con vs Sin)",
            color="promo_flag",
            color_discrete_map={"Con promo": ACCENT_ORANGE, "Sin promo": ACCENT_TURQ},
            text="sales",
        )
        fig.update_traces(texttemplate="%{text:.2s}", textposition="outside", cliponaxis=False)
        fig.update_yaxes(tickformat=".2s")
        apply_plot_style(fig, x_title="", y_title="Ventas medias")
        st.plotly_chart(fig, use_container_width=True)

    # Heatmap día semana vs mes
    if "day_of_week" in df_f.columns and "month" in df_f.columns and "sales" in df_f.columns:
        st.markdown(
            "<div class='small-note'>Cómo leerlo: zonas más intensas = combinaciones día/mes con ventas medias más altas.</div>",
            unsafe_allow_html=True,
        )

        order_en = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        df_h = df_f.copy()
        df_h["day_of_week"] = pd.Categorical(df_h["day_of_week"], categories=order_en, ordered=True)

        pivot = df_h.pivot_table(values="sales", index="day_of_week", columns="month", aggfunc="mean").fillna(0)
        fig = px.imshow(pivot, aspect="auto", title="Heatmap: ventas medias (día semana vs mes)")
        apply_plot_style(fig, x_title="Mes", y_title="Día de la semana")
        st.plotly_chart(fig, use_container_width=True)

    # Festivos
    if "holiday_type" in df_f.columns and "sales" in df_f.columns:
        st.markdown(
            "<div class='small-note'>Cómo leerlo: qué tipo de festivo se asocia a mayor venta media (si aplica en tu dataset).</div>",
            unsafe_allow_html=True,
        )

        hol = df_f.copy()
        hol["holiday_type"] = hol["holiday_type"].fillna("No holiday")
        hol_cmp = (
            hol.groupby("holiday_type", as_index=False)["sales"]
            .mean()
            .sort_values("sales", ascending=False)
            .head(10)
        )

        fig = px.bar(
            hol_cmp,
            x="holiday_type",
            y="sales",
            title="Ventas medias por tipo de festivo (Top 10)",
            color_discrete_sequence=[ACCENT_TURQ],
            text="sales",
        )
        fig.update_traces(texttemplate="%{text:.2s}", textposition="outside", cliponaxis=False)
        fig.update_yaxes(tickformat=".2s")
        apply_plot_style(fig, x_title="Holiday type", y_title="Ventas medias")
        st.plotly_chart(fig, use_container_width=True)

    st.info("💡 Si quieres aún más 'wow': se puede añadir un panel de alertas de outliers (días raros) o un selector global de métrica (ventas/transacciones).")



