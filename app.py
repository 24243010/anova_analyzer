from __future__ import annotations

from datetime import datetime
import os
import tempfile
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from fpdf import FPDF
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="ANOVA Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CONSTANTS
# =========================
COLLEGE_NAME = "National Engineering College, Kovilpatti"
DEPARTMENT = "Department of Artificial Intelligence and Data Science"
INSTRUCTOR_SHORT = "Dr. J.Naskath, B.E., M.E., Ph.D."
INSTRUCTOR_FULL = """Dr. J.Naskath, B.E., M.E., Ph.D.
Assistant Professor
Department of AIDS
National Engineering College (An Autonomous Institution)
K.R. Nagar, Kovilpatti - 628503."""
DEVELOPER = "B.Lavanya - 24243010"
APP_NAME = "ANOVA Analyzer"
TAGLINE = "Analyze • Compare • Visualize • Report"

# =========================
# CSS
# =========================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #ffd6e7 0%, #ffc7dd 35%, #ffbfd8 70%, #ffb3d1 100%);
    color: #4a1630;
}
html, body, [class*="css"] {
    color: #4a1630 !important;
    font-weight: 700;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffc1d6 0%, #ffb3cf 100%) !important;
}
section[data-testid="stSidebar"] * {
    color: #5a1838 !important;
    font-weight: 800 !important;
}
.block-container {
    padding-top: 0.8rem;
    padding-bottom: 2rem;
}
.top-banner {
    background: rgba(255,255,255,0.18);
    border: 2px solid rgba(126, 34, 72, 0.10);
    border-radius: 24px;
    padding: 18px 24px;
    box-shadow: 0 10px 28px rgba(126, 34, 72, 0.08);
    margin-bottom: 1rem;
}
.brand-title {
    text-align: center;
    font-size: 3.5rem;
    font-weight: 900;
    font-style: italic;
    letter-spacing: 1px;
    color: #6f1d46;
    text-shadow: 2px 2px 0 rgba(255,255,255,0.45);
    margin-top: 0.1rem;
    margin-bottom: 0.1rem;
}
.college-line {
    text-align: center;
    font-size: 1.05rem;
    font-weight: 900;
    color: #5a1838;
    margin-top: 0.15rem;
}
.hero-card {
    background: rgba(255,245,249,0.55);
    border: 2px solid rgba(126, 34, 72, 0.12);
    border-radius: 30px;
    padding: 28px 24px;
    box-shadow: 0 12px 30px rgba(126, 34, 72, 0.10);
    margin-top: 0.8rem;
    margin-bottom: 1rem;
}
.welcome-title {
    text-align: center;
    font-size: 2.25rem;
    font-weight: 900;
    color: #7b1e4f;
    margin-top: 0.1rem;
    margin-bottom: 0.45rem;
}
.center-line {
    text-align: center;
    font-size: 1.08rem;
    font-weight: 800;
    color: #5a1838;
    margin-bottom: 0.45rem;
}
.center-line.small {
    font-size: 1rem;
}
.decor-line {
    height: 8px;
    width: 180px;
    margin: 16px auto;
    border-radius: 30px;
    background: linear-gradient(90deg, #ff8fbc, #ffffff, #ff8fbc);
}
.tagline {
    text-align: center;
    font-size: 1.08rem;
    font-weight: 900;
    color: #8b2f58;
    margin-top: 0.7rem;
}
.section-head {
    font-size: 1.35rem;
    font-weight: 900;
    color: #6a163d;
    margin-top: 0.3rem;
    margin-bottom: 0.8rem;
}
.soft-card {
    background: rgba(255, 240, 246, 0.42);
    border: 2px solid rgba(126, 34, 72, 0.15);
    border-radius: 22px;
    padding: 18px;
    box-shadow: 0 10px 24px rgba(126, 34, 72, 0.08);
    margin-bottom: 14px;
}
.feature-card {
    background: rgba(255,255,255,0.35);
    border: 2px solid rgba(126, 34, 72, 0.10);
    border-radius: 24px;
    padding: 18px 16px;
    text-align: center;
    box-shadow: 0 8px 22px rgba(126, 34, 72, 0.08);
    min-height: 150px;
}
.feature-title {
    font-size: 1.1rem;
    font-weight: 900;
    color: #7b1e4f;
    margin-bottom: 0.5rem;
}
.feature-text {
    font-size: 0.98rem;
    font-weight: 700;
    color: #5a1838;
}
.small-highlight {
    background: rgba(255, 245, 249, 0.55);
    border-left: 6px solid #8b2f58;
    border-radius: 12px;
    padding: 12px;
    margin-bottom: 10px;
    font-weight: 800;
}
.success-box {
    padding: 12px;
    border-radius: 12px;
    background: rgba(34, 197, 94, 0.14);
    border: 2px solid rgba(34, 197, 94, 0.28);
    color: #14532d !important;
    font-weight: 900;
}
.warn-box {
    padding: 12px;
    border-radius: 12px;
    background: rgba(245, 158, 11, 0.14);
    border: 2px solid rgba(245, 158, 11, 0.28);
    color: #7c2d12 !important;
    font-weight: 900;
}
.danger-box {
    padding: 12px;
    border-radius: 12px;
    background: rgba(239, 68, 68, 0.14);
    border: 2px solid rgba(239, 68, 68, 0.28);
    color: #7f1d1d !important;
    font-weight: 900;
}
.stButton button, .stDownloadButton button {
    background: #8f2e5a !important;
    color: white !important;
    border-radius: 12px !important;
    border: none !important;
    font-weight: 900 !important;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}
.stTabs [data-baseweb="tab"] {
    background: rgba(255, 240, 246, 0.55);
    border-radius: 12px 12px 0 0;
    font-weight: 900;
    color: #5a1838 !important;
}
table, th, td {
    border: 2px solid #7b1e4f !important;
    border-collapse: collapse !important;
}
th {
    background-color: #ffc3da !important;
    color: #5a1838 !important;
}
td {
    background-color: rgba(255,255,255,0.35) !important;
    color: #4a1630 !important;
}
</style>
""", unsafe_allow_html=True)


# =========================
# GENERAL HELPERS
# =========================
def safe_float_list(text: str) -> List[float]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return values


def dataframe_to_text(df: pd.DataFrame) -> str:
    return df.to_string(index=False)


def add_pdf_text(pdf: FPDF, text: str, font_size: int = 10) -> None:
    pdf.set_font("Arial", size=font_size)
    for line in text.split("\n"):
        pdf.multi_cell(0, 7, txt=line)


def generate_pdf_with_images(report_title: str, report_text: str, image_paths: List[str]) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, report_title, ln=True, align="C")
    pdf.ln(3)
    add_pdf_text(pdf, report_text, font_size=10)

    for path in image_paths:
        if path and os.path.exists(path):
            pdf.add_page()
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Visual Output", ln=True, align="C")
            pdf.ln(4)
            pdf.image(path, x=10, y=25, w=190)

    return pdf.output(dest="S").encode("latin-1", "replace")


def top_header() -> None:
    col1, col2 = st.columns([1, 8])
    with col1:
        try:
            st.image("logo.png", width=90)
        except Exception:
            st.write("")
    with col2:
        st.markdown(
            f"""
            <div class="top-banner">
                <div class="brand-title">{APP_NAME}</div>
                <div class="college-line">{COLLEGE_NAME}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


def center_welcome() -> None:
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="welcome-title">Welcome</div>
            <div class="decor-line"></div>
            <div class="center-line">{TAGLINE}</div>
            <div class="center-line small">Course Instructor: {INSTRUCTOR_SHORT}</div>
            <div class="center-line small">Developed by: {DEVELOPER}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def formula_box(title: str, latex_expr: str) -> None:
    st.markdown(f"**{title}**")
    st.latex(latex_expr)


# =========================
# F TABLE HELPERS
# =========================
def build_f_table(alpha: float = 0.05, max_df2: int = 20, max_df1: int = 10) -> pd.DataFrame:
    rows = []
    for df2 in range(1, max_df2 + 1):
        row = [round(stats.f.ppf(1 - alpha, df1, df2), 4) for df1 in range(1, max_df1 + 1)]
        rows.append(row)
    return pd.DataFrame(rows, index=range(1, max_df2 + 1), columns=range(1, max_df1 + 1))


def style_f_table(df: pd.DataFrame, selected_df1: int, selected_df2: int):
    def highlight(data):
        styled = pd.DataFrame("", index=data.index, columns=data.columns)
        if selected_df2 in data.index:
            styled.loc[selected_df2, :] = "background-color: #ffd7e5;"
        if selected_df1 in data.columns:
            styled.loc[:, selected_df1] = "background-color: #ffc3da;"
        if selected_df2 in data.index and selected_df1 in data.columns:
            styled.loc[selected_df2, selected_df1] = "background-color: #ff7eb6; color: white; font-weight: 900;"
        return styled
    return df.style.apply(highlight, axis=None)


def f_table_snippet(alpha: float, df1: int, df2: int) -> pd.DataFrame:
    max_df2 = max(df2, 6)
    max_df1 = max(df1, 6)
    table = build_f_table(alpha=alpha, max_df2=max_df2, max_df1=max_df1)
    return table.loc[max(1, df2 - 2): df2 + 2, max(1, df1 - 2): df1 + 2]


# =========================
# ONE-WAY ANOVA
# =========================
def one_way_manual_input() -> Tuple[float, List[Tuple[str, str]]]:
    st.markdown('<div class="section-head">Manual Data Entry</div>', unsafe_allow_html=True)
    group_count = st.number_input("Number of Groups", min_value=2, max_value=8, value=3, step=1)
    alpha = st.selectbox("Alpha (Significance Level)", [0.10, 0.05, 0.01], index=1)

    defaults = [
        "25, 33, 45, 28",
        "35, 39, 38, 32",
        "33, 32, 31, 39",
        "29, 30, 32, 31"
    ]

    groups = []
    cols = st.columns(min(int(group_count), 4))
    for i in range(int(group_count)):
        with cols[i % len(cols)]:
            txt = st.text_area(
                f"Group {i+1} Values",
                value=defaults[i] if i < len(defaults) else "",
                height=110,
                key=f"ow_group_{i}"
            )
            groups.append((f"Group {i+1}", txt))
    return alpha, groups


def one_way_csv_input() -> Tuple[float, pd.DataFrame | None, str | None, str | None]:
    st.markdown('<div class="section-head">CSV Upload</div>', unsafe_allow_html=True)
    alpha = st.selectbox("Alpha (Significance Level)", [0.10, 0.05, 0.01], index=1, key="ow_csv_alpha")
    uploaded = st.file_uploader("Upload CSV file", type=["csv"], key="ow_csv")

    if uploaded is None:
        st.info("CSV should contain one grouping column and one numeric value column.")
        return alpha, None, None, None

    df = pd.read_csv(uploaded)
    st.dataframe(df.head(), use_container_width=True)

    group_col = st.selectbox("Select Group Column", df.columns, key="ow_group_col")
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    if not numeric_cols:
        st.error("No numeric column found in CSV.")
        return alpha, None, None, None

    value_col = st.selectbox("Select Value Column", numeric_cols, key="ow_value_col")
    return alpha, df, group_col, value_col


def compute_one_way(groups_dict: Dict[str, List[float]], alpha: float) -> dict:
    names = list(groups_dict.keys())
    arrays = [np.array(values, dtype=float) for values in groups_dict.values()]

    if any(len(arr) < 2 for arr in arrays):
        raise ValueError("Each group must contain at least 2 values.")

    n_i = [len(arr) for arr in arrays]
    sums = [np.sum(arr) for arr in arrays]
    means = [np.mean(arr) for arr in arrays]
    variances = [np.var(arr, ddof=1) for arr in arrays]

    all_values = np.concatenate(arrays)
    N = len(all_values)
    k = len(arrays)
    grand_mean = np.mean(all_values)

    ss_between = sum(n_i[i] * (means[i] - grand_mean) ** 2 for i in range(k))
    ss_within = sum(np.sum((arrays[i] - means[i]) ** 2) for i in range(k))
    ss_total = np.sum((all_values - grand_mean) ** 2)

    df_between = k - 1
    df_within = N - k
    df_total = N - 1

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    f_cal = ms_between / ms_within
    p_value = 1 - stats.f.cdf(f_cal, df_between, df_within)
    f_critical = stats.f.ppf(1 - alpha, df_between, df_within)

    summary_df = pd.DataFrame({
        "Group": names,
        "Samples": n_i,
        "Sum": np.round(sums, 4),
        "Average": np.round(means, 4),
        "Variance": np.round(variances, 4)
    })

    anova_df = pd.DataFrame({
        "Source of Variation": ["Between Groups", "Within Groups", "Total"],
        "Sum of Squares": [round(ss_between, 4), round(ss_within, 4), round(ss_total, 4)],
        "Degrees of Freedom": [df_between, df_within, df_total],
        "Mean Square": [round(ms_between, 4), round(ms_within, 4), ""],
        "F": [round(f_cal, 4), "", ""],
        "p-value": [round(p_value, 4), "", ""]
    })

    hypothesis = {
        "H0": "All group means are equal.",
        "H1": "At least one group mean is different."
    }

    result_table = pd.DataFrame({
        "Metric": ["F Calculated", "F Critical", "p-value", "Decision"],
        "Value": [
            round(f_cal, 4),
            round(f_critical, 4),
            round(p_value, 4),
            "Reject H0" if f_cal > f_critical else "Fail to Reject H0"
        ]
    })

    return {
        "summary_df": summary_df,
        "anova_df": anova_df,
        "names": names,
        "arrays": arrays,
        "grand_mean": grand_mean,
        "ss_between": ss_between,
        "ss_within": ss_within,
        "ss_total": ss_total,
        "df_between": df_between,
        "df_within": df_within,
        "df_total": df_total,
        "ms_between": ms_between,
        "ms_within": ms_within,
        "f_cal": f_cal,
        "p_value": p_value,
        "f_critical": f_critical,
        "alpha": alpha,
        "hypothesis": hypothesis,
        "result_table": result_table
    }


def one_way_plotly_figures(result: dict):
    names = result["names"]
    arrays = result["arrays"]

    long_df = pd.DataFrame({
        "Group": np.concatenate([[names[i]] * len(arrays[i]) for i in range(len(names))]),
        "Value": np.concatenate(arrays)
    })

    fig1 = px.box(
        long_df,
        x="Group",
        y="Value",
        points="all",
        title="Box Plot of Groups",
        color="Group",
        color_discrete_sequence=px.colors.qualitative.Bold,
        hover_data=["Value"]
    )
    fig1.update_layout(title_x=0.35)

    mean_df = long_df.groupby("Group", as_index=False)["Value"].mean()
    fig2 = px.bar(
        mean_df,
        x="Group",
        y="Value",
        text_auto=".2f",
        title="Group Means",
        color="Group",
        color_discrete_sequence=px.colors.qualitative.Set2,
        hover_data={"Value": ":.4f"}
    )
    fig2.update_layout(title_x=0.4)

    x = np.linspace(0.001, max(result["f_critical"] * 1.8, result["f_cal"] * 1.8, 5), 500)
    y = stats.f.pdf(x, result["df_between"], result["df_within"])

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="lines",
        name="F Distribution",
        line=dict(color="#7c3aed", width=4),
        hovertemplate="F value=%{x:.4f}<br>Density=%{y:.4f}<extra></extra>"
    ))
    fig3.add_vline(x=result["f_critical"], line_dash="dash", line_width=3, line_color="red")
    fig3.add_vline(x=result["f_cal"], line_width=3, line_color="green")
    fig3.add_annotation(x=result["f_critical"], y=max(y) * 0.85, text=f"F Critical = {result['f_critical']:.4f}", showarrow=True)
    fig3.add_annotation(x=result["f_cal"], y=max(y) * 0.55, text=f"F Calculated = {result['f_cal']:.4f}", showarrow=True)
    fig3.update_layout(
        title="F Distribution Curve",
        title_x=0.38,
        xaxis_title="F Value",
        yaxis_title="Density"
    )

    return fig1, fig2, fig3


def save_one_way_matplotlib_images(result: dict, folder: str) -> List[str]:
    names = result["names"]
    arrays = result["arrays"]
    image_paths = []

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(arrays, labels=names, patch_artist=True)
    ax.set_title("Box Plot of Groups")
    ax.set_ylabel("Values")
    ax.grid(alpha=0.3)
    p1 = os.path.join(folder, "ow_box.png")
    fig.savefig(p1, bbox_inches="tight", dpi=200)
    plt.close(fig)
    image_paths.append(p1)

    fig, ax = plt.subplots(figsize=(10, 5))
    means = [np.mean(a) for a in arrays]
    ax.bar(names, means)
    for i, val in enumerate(means):
        ax.text(i, val, f"{val:.2f}", ha="center", va="bottom")
    ax.set_title("Group Means")
    ax.set_ylabel("Mean")
    ax.grid(axis="y", alpha=0.3)
    p2 = os.path.join(folder, "ow_bar.png")
    fig.savefig(p2, bbox_inches="tight", dpi=200)
    plt.close(fig)
    image_paths.append(p2)

    x = np.linspace(0.001, max(result["f_critical"] * 1.8, result["f_cal"] * 1.8, 5), 500)
    y = stats.f.pdf(x, result["df_between"], result["df_within"])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, linewidth=2)
    ax.axvline(result["f_critical"], linestyle="--", linewidth=2)
    ax.axvline(result["f_cal"], linewidth=2)
    ax.set_title("F Distribution Curve")
    ax.set_xlabel("F Value")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.3)
    p3 = os.path.join(folder, "ow_fcurve.png")
    fig.savefig(p3, bbox_inches="tight", dpi=200)
    plt.close(fig)
    image_paths.append(p3)

    return image_paths


def make_one_way_report(result: dict) -> str:
    decision = "Reject Null Hypothesis" if result["f_cal"] > result["f_critical"] else "Fail to Reject Null Hypothesis"
    interpretation = (
        "There is a significant difference among the group means."
        if result["f_cal"] > result["f_critical"]
        else "There is no significant difference among the group means."
    )
    snippet = f_table_snippet(result["alpha"], result["df_between"], result["df_within"])

    return f"""{APP_NAME}
{COLLEGE_NAME}
{DEPARTMENT}
Course Instructor:
{INSTRUCTOR_FULL}
Developed By:
{DEVELOPER}
Generated On: {datetime.now().strftime("%d-%m-%Y %H:%M:%S")}

HYPOTHESIS
H0: {result["hypothesis"]["H0"]}
H1: {result["hypothesis"]["H1"]}

ALPHA
{result["alpha"]}

GROUP SUMMARY
{dataframe_to_text(result["summary_df"])}

GRAND MEAN
{round(result["grand_mean"], 4)}

ANOVA TABLE
{dataframe_to_text(result["anova_df"])}

CRITICAL VALUE TABLE SNIPPET
{dataframe_to_text(snippet.reset_index().rename(columns={"index": "df2"}))}

VISUALS INCLUDED
1. Box Plot of Groups
2. Group Means Bar Chart
3. F Distribution Curve

FINAL RESULT TABLE
{dataframe_to_text(result["result_table"])}

DECISION
{decision}

INTERPRETATION
{interpretation}
"""


# =========================
# TWO-WAY ANOVA
# =========================
def two_way_manual_input() -> Tuple[float, str, str, str]:
    st.markdown('<div class="section-head">Manual Matrix Entry</div>', unsafe_allow_html=True)
    alpha = st.selectbox("Alpha (Significance Level)", [0.10, 0.05, 0.01], index=1, key="tw_alpha")
    row_labels = st.text_input("Row Factor Levels", "Machine 1, Machine 2, Machine 3, Machine 4")
    col_labels = st.text_input("Column Factor Levels", "Auto 1, Auto 2, Auto 3")
    matrix_text = st.text_area(
        "Enter Matrix Row by Row",
        value="28,25,30\n20,29,27\n28,27,31\n36,23,28",
        height=180
    )
    return alpha, row_labels, col_labels, matrix_text


def two_way_csv_input() -> Tuple[float, pd.DataFrame | None, str | None, str | None, str | None]:
    st.markdown('<div class="section-head">CSV Upload</div>', unsafe_allow_html=True)
    alpha = st.selectbox("Alpha (Significance Level)", [0.10, 0.05, 0.01], index=1, key="tw_csv_alpha")
    uploaded = st.file_uploader("Upload CSV file", type=["csv"], key="tw_csv")

    if uploaded is None:
        st.info("CSV should contain Factor A, Factor B, and one numeric response column.")
        return alpha, None, None, None, None

    df = pd.read_csv(uploaded)
    st.dataframe(df.head(), use_container_width=True)

    factor_a = st.selectbox("Select Factor A Column", df.columns, key="tw_factor_a")
    factor_b = st.selectbox("Select Factor B Column", df.columns, key="tw_factor_b")
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    if not numeric_cols:
        st.error("No numeric response column found.")
        return alpha, None, None, None, None

    value_col = st.selectbox("Select Response Column", numeric_cols, key="tw_value_col")
    return alpha, df, factor_a, factor_b, value_col


def build_two_way_df_manual(row_labels: str, col_labels: str, matrix_text: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = [r.strip() for r in row_labels.split(",") if r.strip()]
    cols = [c.strip() for c in col_labels.split(",") if c.strip()]
    lines = [line.strip() for line in matrix_text.strip().splitlines() if line.strip()]

    if len(lines) != len(rows):
        raise ValueError("Number of matrix rows must match row labels.")

    data_records = []
    matrix = []
    for i, line in enumerate(lines):
        values = [float(x.strip()) for x in line.split(",") if x.strip()]
        if len(values) != len(cols):
            raise ValueError(f"Row {i+1} must contain exactly {len(cols)} values.")
        matrix.append(values)
        for j, val in enumerate(values):
            data_records.append({"FactorA": rows[i], "FactorB": cols[j], "Value": val})

    matrix_df = pd.DataFrame(matrix, index=rows, columns=cols)
    long_df = pd.DataFrame(data_records)
    return long_df, matrix_df


def compute_two_way(df_long: pd.DataFrame, alpha: float) -> dict:
    model = ols("Value ~ C(FactorA) + C(FactorB)", data=df_long).fit()

    anova = anova_lm(model, typ=2).reset_index().rename(columns={"index": "Source"})
    anova["sum_sq"] = anova["sum_sq"].round(4)
    anova["mean_sq"] = (anova["sum_sq"] / anova["df"]).round(4)
    anova["F"] = anova["F"].round(4)
    anova["PR(>F)"] = anova["PR(>F)"].round(4)

    row_summary = df_long.groupby("FactorA")["Value"].agg(["count", "sum", "mean", "var"]).reset_index()
    row_summary.columns = ["Factor A", "Samples", "Sum", "Average", "Variance"]
    row_summary = row_summary.round(4)

    col_summary = df_long.groupby("FactorB")["Value"].agg(["count", "sum", "mean", "var"]).reset_index()
    col_summary.columns = ["Factor B", "Samples", "Sum", "Average", "Variance"]
    col_summary = col_summary.round(4)

    grand_mean = round(df_long["Value"].mean(), 4)
    total_variance = round(df_long["Value"].var(ddof=1), 4)

    error_row = anova[anova["Source"] == "Residual"]
    if error_row.empty:
        raise ValueError("Residual row not found. Check your data.")

    df_error = int(round(error_row["df"].iloc[0]))

    critical_values = {}
    for _, row in anova.iterrows():
        if row["Source"] != "Residual":
            df_num = int(round(row["df"]))
            critical_values[row["Source"]] = round(stats.f.ppf(1 - alpha, df_num, df_error), 4)

    result_rows = []
    for _, row in anova.iterrows():
        if row["Source"] != "Residual":
            src = row["Source"]
            decision = "Reject H0" if row["F"] > critical_values[src] else "Fail to Reject H0"
            result_rows.append([src, row["F"], critical_values[src], row["PR(>F)"], decision])

    result_table = pd.DataFrame(
        result_rows,
        columns=["Factor", "F Calculated", "F Critical", "p-value", "Decision"]
    )

    return {
        "anova_df": anova,
        "row_summary": row_summary,
        "col_summary": col_summary,
        "grand_mean": grand_mean,
        "total_variance": total_variance,
        "df_error": df_error,
        "critical_values": critical_values,
        "alpha": alpha,
        "result_table": result_table
    }


def two_way_plotly_figures(df_long: pd.DataFrame):
    pivot_df = df_long.pivot(index="FactorA", columns="FactorB", values="Value")

    fig1 = px.imshow(
        pivot_df,
        text_auto=True,
        aspect="auto",
        title="Heatmap of Two-Way Data",
        color_continuous_scale="RdPu"
    )
    fig1.update_layout(title_x=0.35)

    mean_df = df_long.groupby(["FactorA", "FactorB"], as_index=False)["Value"].mean()
    fig2 = px.line(
        mean_df,
        x="FactorB",
        y="Value",
        color="FactorA",
        markers=True,
        title="Interaction Plot",
        color_discrete_sequence=px.colors.qualitative.Vivid,
        hover_data={"Value": ":.4f"}
    )
    fig2.update_layout(title_x=0.4)

    fig3 = px.bar(
        mean_df,
        x="FactorB",
        y="Value",
        color="FactorA",
        barmode="group",
        title="Grouped Bar Chart",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig3.update_layout(title_x=0.4)

    return fig1, fig2, fig3


def save_two_way_matplotlib_images(df_long: pd.DataFrame, folder: str) -> List[str]:
    image_paths = []
    pivot_df = df_long.pivot(index="FactorA", columns="FactorB", values="Value")

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(pivot_df.values, aspect="auto")
    ax.set_title("Heatmap of Two-Way Data")
    ax.set_xticks(range(len(pivot_df.columns)))
    ax.set_xticklabels(pivot_df.columns, rotation=30)
    ax.set_yticks(range(len(pivot_df.index)))
    ax.set_yticklabels(pivot_df.index)
    plt.colorbar(im, ax=ax)
    p1 = os.path.join(folder, "tw_heatmap.png")
    fig.savefig(p1, bbox_inches="tight", dpi=200)
    plt.close(fig)
    image_paths.append(p1)

    fig, ax = plt.subplots(figsize=(9, 5))
    for factor_a in pivot_df.index:
        ax.plot(pivot_df.columns, pivot_df.loc[factor_a].values, marker="o", linewidth=2, label=factor_a)
    ax.set_title("Interaction Plot")
    ax.set_xlabel("Factor B")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(alpha=0.3)
    p2 = os.path.join(folder, "tw_interaction.png")
    fig.savefig(p2, bbox_inches="tight", dpi=200)
    plt.close(fig)
    image_paths.append(p2)

    fig, ax = plt.subplots(figsize=(9, 5))
    mean_df = df_long.groupby(["FactorA", "FactorB"], as_index=False)["Value"].mean()
    factors_a = mean_df["FactorA"].unique()
    factors_b = mean_df["FactorB"].unique()
    x = np.arange(len(factors_b))
    width = 0.8 / max(len(factors_a), 1)
    for i, fa in enumerate(factors_a):
        vals = mean_df[mean_df["FactorA"] == fa]["Value"].values
        ax.bar(x + i * width, vals, width=width, label=fa)
    ax.set_xticks(x + width * (len(factors_a) - 1) / 2)
    ax.set_xticklabels(factors_b)
    ax.set_title("Grouped Bar Chart")
    ax.set_ylabel("Mean Value")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    p3 = os.path.join(folder, "tw_bar.png")
    fig.savefig(p3, bbox_inches="tight", dpi=200)
    plt.close(fig)
    image_paths.append(p3)

    return image_paths


def make_two_way_report(result: dict) -> str:
    lines = []
    for src, crit in result["critical_values"].items():
        lines.append(f"{src}: {crit}")

    first_source = list(result["critical_values"].keys())[0]
    first_df1 = int(round(result["anova_df"][result["anova_df"]["Source"] == first_source]["df"].iloc[0]))
    snippet = f_table_snippet(result["alpha"], first_df1, result["df_error"])

    return f"""{APP_NAME}
{COLLEGE_NAME}
{DEPARTMENT}
Course Instructor:
{INSTRUCTOR_FULL}
Developed By:
{DEVELOPER}
Generated On: {datetime.now().strftime("%d-%m-%Y %H:%M:%S")}

HYPOTHESIS
For each factor:
H0: Means are equal
H1: At least one mean is different

ALPHA
{result["alpha"]}

FACTOR A SUMMARY
{dataframe_to_text(result["row_summary"])}

FACTOR B SUMMARY
{dataframe_to_text(result["col_summary"])}

GRAND MEAN
{result["grand_mean"]}

TOTAL VARIANCE
{result["total_variance"]}

ANOVA TABLE
{dataframe_to_text(result["anova_df"])}

CRITICAL VALUES
{chr(10).join(lines)}

CRITICAL VALUE TABLE SNIPPET
{dataframe_to_text(snippet.reset_index().rename(columns={"index": "df2"}))}

VISUALS INCLUDED
1. Heatmap
2. Interaction Plot
3. Grouped Bar Chart

FINAL RESULT TABLE
{dataframe_to_text(result["result_table"])}
"""


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("## Navigation")
    page = st.radio("Go to", ["Home", "ANOVA Analysis"])
    st.markdown("---")
    st.markdown("### Main Features")
    st.markdown("""
- One-Way ANOVA  
- Two-Way ANOVA  
- Manual Entry  
- CSV Upload  
- Formula Display  
- F Critical Table  
- Colorful Visualisation  
- Report Download  
""")


# =========================
# HOME PAGE
# =========================
if page == "Home":
    top_header()
    center_welcome()

    st.markdown('<div class="section-head">Highlights</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">One-Way ANOVA</div>
            <div class="feature-text">
                Compare multiple group means with formulas, ANOVA table, and critical value support.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">Two-Way ANOVA</div>
            <div class="feature-text">
                Analyze two factors with summaries, interaction view, and statistical output.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">Visual Reports</div>
            <div class="feature-text">
                Interactive graphs, critical value table, hover values, and downloadable report.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="soft-card">
        <div class="section-head">App Workflow</div>
        <div class="feature-text">
            1. Choose One-Way ANOVA or Two-Way ANOVA<br>
            2. Enter data manually or upload CSV file<br>
            3. View formulas, calculations, and ANOVA table<br>
            4. Compare F Calculated with F Critical value<br>
            5. Explore interactive visualisation with hover values<br>
            6. Generate and download report
        </div>
    </div>
    """, unsafe_allow_html=True)

# =========================
# ANALYSIS PAGE
# =========================
else:
    top_header()

    analysis_type = st.radio(
        "Choose Analysis Type",
        ["One-Way ANOVA", "Two-Way ANOVA"],
        horizontal=True
    )

    input_mode = st.radio(
        "Choose Input Method",
        ["Manual Entry", "CSV Upload"],
        horizontal=True
    )

    st.markdown("---")

    if analysis_type == "One-Way ANOVA":
        if input_mode == "Manual Entry":
            alpha, group_inputs = one_way_manual_input()

            if st.button("Run One-Way ANOVA", use_container_width=True):
                try:
                    groups_dict = {}
                    for name, txt in group_inputs:
                        vals = safe_float_list(txt)
                        if len(vals) == 0:
                            raise ValueError(f"{name} is empty.")
                        groups_dict[name] = vals

                    result = compute_one_way(groups_dict, alpha)
                    fig1, fig2, fig3 = one_way_plotly_figures(result)

                    calc_tab, visual_tab, report_tab = st.tabs(["Calculation", "Visualisation", "Report"])

                    with calc_tab:
                        left, right = st.columns([3, 1])

                        with left:
                            st.markdown('<div class="section-head">Hypothesis</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="small-highlight">H0: {result["hypothesis"]["H0"]}</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="small-highlight">H1: {result["hypothesis"]["H1"]}</div>', unsafe_allow_html=True)

                            st.markdown('<div class="section-head">Group Summary</div>', unsafe_allow_html=True)
                            st.table(result["summary_df"])

                            st.markdown('<div class="section-head">Formulas</div>', unsafe_allow_html=True)
                            formula_box("Grand Mean", r"\bar{Y}_{grand} = \frac{\sum Y}{N}")
                            formula_box("Sum of Squares Between", r"SSB = \sum n_i(\bar{Y}_i - \bar{Y}_{grand})^2")
                            formula_box("Sum of Squares Within", r"SSW = \sum\sum (Y_{ij} - \bar{Y}_i)^2")
                            formula_box("Total Sum of Squares", r"SST = SSB + SSW")
                            formula_box("Mean Squares", r"MSB = \frac{SSB}{df_B}, \quad MSW = \frac{SSW}{df_W}")
                            formula_box("F Ratio", r"F = \frac{MSB}{MSW}")

                            st.markdown('<div class="section-head">Calculated Values</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="small-highlight">Grand Mean: {result["grand_mean"]:.4f}</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="small-highlight">SS Between: {result["ss_between"]:.4f}</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="small-highlight">SS Within: {result["ss_within"]:.4f}</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="small-highlight">SS Total: {result["ss_total"]:.4f}</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="small-highlight">F Calculated: {result["f_cal"]:.4f}</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="small-highlight">p-value: {result["p_value"]:.4f}</div>', unsafe_allow_html=True)

                            st.markdown('<div class="section-head">ANOVA Table</div>', unsafe_allow_html=True)
                            st.table(result["anova_df"])

                            st.markdown('<div class="section-head">Final Result Table</div>', unsafe_allow_html=True)
                            st.table(result["result_table"])

                            st.markdown('<div class="section-head">Decision</div>', unsafe_allow_html=True)
                            if result["f_cal"] > result["f_critical"]:
                                st.markdown(
                                    f'<div class="success-box">Since F Calculated ({result["f_cal"]:.4f}) > F Critical ({result["f_critical"]:.4f}), Reject Null Hypothesis.</div>',
                                    unsafe_allow_html=True
                                )
                            else:
                                st.markdown(
                                    f'<div class="warn-box">Since F Calculated ({result["f_cal"]:.4f}) ≤ F Critical ({result["f_critical"]:.4f}), Fail to Reject Null Hypothesis.</div>',
                                    unsafe_allow_html=True
                                )

                        with right:
                            st.markdown('<div class="section-head">F Critical Table</div>', unsafe_allow_html=True)
                            st.write(f"Alpha = {alpha}")
                            st.write(f"Numerator df = {result['df_between']}")
                            st.write(f"Denominator df = {result['df_within']}")
                            st.write(f"Selected Critical Value = {result['f_critical']:.4f}")
                            f_table = build_f_table(alpha=alpha)
                            st.dataframe(
                                style_f_table(f_table, result["df_between"], result["df_within"]),
                                use_container_width=True,
                                height=550
                            )

                    with visual_tab:
                        st.plotly_chart(fig1, use_container_width=True, key="ow_visual_box")
                        st.plotly_chart(fig2, use_container_width=True, key="ow_visual_bar")
                        st.plotly_chart(fig3, use_container_width=True, key="ow_visual_fcurve")

                    with report_tab:
                        report_text = make_one_way_report(result)
                        st.text_area("Report Preview", report_text, height=420)

                        st.markdown('<div class="section-head">Critical Value Table Snippet</div>', unsafe_allow_html=True)
                        st.table(f_table_snippet(result["alpha"], result["df_between"], result["df_within"]))

                        st.markdown('<div class="section-head">Visuals</div>', unsafe_allow_html=True)
                        st.plotly_chart(fig1, use_container_width=True, key="ow_report_box")
                        st.plotly_chart(fig2, use_container_width=True, key="ow_report_bar")
                        st.plotly_chart(fig3, use_container_width=True, key="ow_report_fcurve")

                        st.markdown('<div class="section-head">Final Result Table</div>', unsafe_allow_html=True)
                        st.table(result["result_table"])

                        with tempfile.TemporaryDirectory() as tmpdir:
                            image_paths = save_one_way_matplotlib_images(result, tmpdir)
                            pdf_bytes = generate_pdf_with_images("One-Way ANOVA Report", report_text, image_paths)

                            c1, c2 = st.columns(2)
                            with c1:
                                st.download_button(
                                    "Download TXT Report",
                                    report_text,
                                    file_name="one_way_anova_report.txt",
                                    mime="text/plain",
                                    use_container_width=True
                                )
                            with c2:
                                st.download_button(
                                    "Download PDF Report",
                                    pdf_bytes,
                                    file_name="one_way_anova_report.pdf",
                                    mime="application/pdf",
                                    use_container_width=True
                                )

                except Exception as e:
                    st.markdown(f'<div class="danger-box">Error: {str(e)}</div>', unsafe_allow_html=True)

        else:
            alpha, df_csv, group_col, value_col = one_way_csv_input()

            if df_csv is not None and st.button("Run One-Way ANOVA from CSV", use_container_width=True):
                try:
                    clean_df = df_csv[[group_col, value_col]].dropna().copy()
                    groups_dict = {
                        str(g): list(clean_df.loc[clean_df[group_col] == g, value_col].astype(float).values)
                        for g in clean_df[group_col].unique()
                    }

                    result = compute_one_way(groups_dict, alpha)
                    fig1, fig2, fig3 = one_way_plotly_figures(result)

                    calc_tab, visual_tab, report_tab = st.tabs(["Calculation", "Visualisation", "Report"])

                    with calc_tab:
                        left, right = st.columns([3, 1])

                        with left:
                            st.markdown('<div class="section-head">Group Summary</div>', unsafe_allow_html=True)
                            st.table(result["summary_df"])

                            st.markdown('<div class="section-head">ANOVA Table</div>', unsafe_allow_html=True)
                            st.table(result["anova_df"])

                            st.markdown('<div class="section-head">Final Result</div>', unsafe_allow_html=True)
                            st.table(result["result_table"])

                        with right:
                            st.markdown('<div class="section-head">F Critical Table</div>', unsafe_allow_html=True)
                            st.write(f"Alpha = {alpha}")
                            st.write(f"Numerator df = {result['df_between']}")
                            st.write(f"Denominator df = {result['df_within']}")
                            st.write(f"Selected Critical Value = {result['f_critical']:.4f}")
                            f_table = build_f_table(alpha=alpha)
                            st.dataframe(
                                style_f_table(f_table, result["df_between"], result["df_within"]),
                                use_container_width=True,
                                height=550
                            )

                    with visual_tab:
                        st.plotly_chart(fig1, use_container_width=True, key="ow_csv_visual_box")
                        st.plotly_chart(fig2, use_container_width=True, key="ow_csv_visual_bar")
                        st.plotly_chart(fig3, use_container_width=True, key="ow_csv_visual_fcurve")

                    with report_tab:
                        report_text = make_one_way_report(result)
                        st.text_area("Report Preview", report_text, height=420)
                        st.table(f_table_snippet(result["alpha"], result["df_between"], result["df_within"]))
                        st.table(result["result_table"])
                        st.plotly_chart(fig1, use_container_width=True, key="ow_csv_report_box")
                        st.plotly_chart(fig2, use_container_width=True, key="ow_csv_report_bar")
                        st.plotly_chart(fig3, use_container_width=True, key="ow_csv_report_fcurve")

                        with tempfile.TemporaryDirectory() as tmpdir:
                            image_paths = save_one_way_matplotlib_images(result, tmpdir)
                            pdf_bytes = generate_pdf_with_images("One-Way ANOVA Report", report_text, image_paths)
                            st.download_button(
                                "Download PDF Report",
                                pdf_bytes,
                                file_name="one_way_anova_report.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )

                except Exception as e:
                    st.markdown(f'<div class="danger-box">Error: {str(e)}</div>', unsafe_allow_html=True)

    else:
        if input_mode == "Manual Entry":
            alpha, row_labels, col_labels, matrix_text = two_way_manual_input()

            if st.button("Run Two-Way ANOVA", use_container_width=True):
                try:
                    df_long, matrix_df = build_two_way_df_manual(row_labels, col_labels, matrix_text)
                    result = compute_two_way(df_long, alpha)
                    fig1, fig2, fig3 = two_way_plotly_figures(df_long)

                    calc_tab, visual_tab, report_tab = st.tabs(["Calculation", "Visualisation", "Report"])

                    with calc_tab:
                        left, right = st.columns([3, 1])

                        with left:
                            st.markdown('<div class="section-head">Hypothesis</div>', unsafe_allow_html=True)
                            st.markdown('<div class="small-highlight">H0: Means are equal</div>', unsafe_allow_html=True)
                            st.markdown('<div class="small-highlight">H1: At least one mean is different</div>', unsafe_allow_html=True)

                            st.markdown('<div class="section-head">Input Matrix</div>', unsafe_allow_html=True)
                            st.table(matrix_df)

                            st.markdown('<div class="section-head">Factor A Summary</div>', unsafe_allow_html=True)
                            st.table(result["row_summary"])

                            st.markdown('<div class="section-head">Factor B Summary</div>', unsafe_allow_html=True)
                            st.table(result["col_summary"])

                            st.markdown('<div class="section-head">Formula View</div>', unsafe_allow_html=True)
                            formula_box("Two-Way ANOVA Model", r"Y_{ij} = \mu + \alpha_i + \beta_j + \epsilon_{ij}")

                            st.markdown('<div class="section-head">ANOVA Table</div>', unsafe_allow_html=True)
                            st.table(result["anova_df"])

                            st.markdown('<div class="section-head">Final Result Table</div>', unsafe_allow_html=True)
                            st.table(result["result_table"])

                        with right:
                            st.markdown('<div class="section-head">F Critical Table</div>', unsafe_allow_html=True)
                            f_table = build_f_table(alpha=alpha)
                            non_resid = result["anova_df"][result["anova_df"]["Source"] != "Residual"]
                            if not non_resid.empty:
                                first_df1 = int(round(non_resid.iloc[0]["df"]))
                                st.dataframe(
                                    style_f_table(f_table, first_df1, result["df_error"]),
                                    use_container_width=True,
                                    height=550
                                )
                                for src, crit in result["critical_values"].items():
                                    st.write(f"{src} → {crit}")

                    with visual_tab:
                        st.plotly_chart(fig1, use_container_width=True, key="tw_visual_heatmap")
                        st.plotly_chart(fig2, use_container_width=True, key="tw_visual_interaction")
                        st.plotly_chart(fig3, use_container_width=True, key="tw_visual_bar")

                    with report_tab:
                        report_text = make_two_way_report(result)
                        st.text_area("Report Preview", report_text, height=420)

                        first_source = list(result["critical_values"].keys())[0]
                        first_df1 = int(round(result["anova_df"][result["anova_df"]["Source"] == first_source]["df"].iloc[0]))

                        st.markdown('<div class="section-head">Critical Value Table Snippet</div>', unsafe_allow_html=True)
                        st.table(f_table_snippet(result["alpha"], first_df1, result["df_error"]))

                        st.markdown('<div class="section-head">Visuals</div>', unsafe_allow_html=True)
                        st.plotly_chart(fig1, use_container_width=True, key="tw_report_heatmap")
                        st.plotly_chart(fig2, use_container_width=True, key="tw_report_interaction")
                        st.plotly_chart(fig3, use_container_width=True, key="tw_report_bar")

                        st.markdown('<div class="section-head">Final Result Table</div>', unsafe_allow_html=True)
                        st.table(result["result_table"])

                        with tempfile.TemporaryDirectory() as tmpdir:
                            image_paths = save_two_way_matplotlib_images(df_long, tmpdir)
                            pdf_bytes = generate_pdf_with_images("Two-Way ANOVA Report", report_text, image_paths)

                            c1, c2 = st.columns(2)
                            with c1:
                                st.download_button(
                                    "Download TXT Report",
                                    report_text,
                                    file_name="two_way_anova_report.txt",
                                    mime="text/plain",
                                    use_container_width=True
                                )
                            with c2:
                                st.download_button(
                                    "Download PDF Report",
                                    pdf_bytes,
                                    file_name="two_way_anova_report.pdf",
                                    mime="application/pdf",
                                    use_container_width=True
                                )

                except Exception as e:
                    st.markdown(f'<div class="danger-box">Error: {str(e)}</div>', unsafe_allow_html=True)

        else:
            alpha, df_csv, factor_a, factor_b, value_col = two_way_csv_input()

            if df_csv is not None and st.button("Run Two-Way ANOVA from CSV", use_container_width=True):
                try:
                    df_long = df_csv[[factor_a, factor_b, value_col]].dropna().copy()
                    df_long.columns = ["FactorA", "FactorB", "Value"]
                    df_long["Value"] = pd.to_numeric(df_long["Value"], errors="coerce")
                    df_long = df_long.dropna()

                    result = compute_two_way(df_long, alpha)
                    fig1, fig2, fig3 = two_way_plotly_figures(df_long)

                    calc_tab, visual_tab, report_tab = st.tabs(["Calculation", "Visualisation", "Report"])

                    with calc_tab:
                        left, right = st.columns([3, 1])

                        with left:
                            st.markdown('<div class="section-head">Factor A Summary</div>', unsafe_allow_html=True)
                            st.table(result["row_summary"])

                            st.markdown('<div class="section-head">Factor B Summary</div>', unsafe_allow_html=True)
                            st.table(result["col_summary"])

                            st.markdown('<div class="section-head">ANOVA Table</div>', unsafe_allow_html=True)
                            st.table(result["anova_df"])

                            st.markdown('<div class="section-head">Final Result</div>', unsafe_allow_html=True)
                            st.table(result["result_table"])

                        with right:
                            st.markdown('<div class="section-head">F Critical Table</div>', unsafe_allow_html=True)
                            f_table = build_f_table(alpha=alpha)
                            non_resid = result["anova_df"][result["anova_df"]["Source"] != "Residual"]
                            if not non_resid.empty:
                                df1 = int(round(non_resid.iloc[0]["df"]))
                                df2 = result["df_error"]

                                st.write(f"df1 = {df1}")
                                st.write(f"df2 = {df2}")

                                st.dataframe(
                                    style_f_table(f_table, df1, df2),
                                    use_container_width=True,
                                    height=500
                                )

                                for src, crit in result["critical_values"].items():
                                    st.write(f"{src} → {crit}")

                    with visual_tab:
                        st.plotly_chart(fig1, use_container_width=True, key="tw_csv_visual_heatmap")
                        st.plotly_chart(fig2, use_container_width=True, key="tw_csv_visual_interaction")
                        st.plotly_chart(fig3, use_container_width=True, key="tw_csv_visual_bar")

                    with report_tab:
                        report_text = make_two_way_report(result)
                        st.text_area("Report Preview", report_text, height=420)

                        first_source = list(result["critical_values"].keys())[0]
                        first_df1 = int(round(result["anova_df"][result["anova_df"]["Source"] == first_source]["df"].iloc[0]))

                        st.table(f_table_snippet(result["alpha"], first_df1, result["df_error"]))
                        st.table(result["result_table"])
                        st.plotly_chart(fig1, use_container_width=True, key="tw_csv_report_heatmap")
                        st.plotly_chart(fig2, use_container_width=True, key="tw_csv_report_interaction")
                        st.plotly_chart(fig3, use_container_width=True, key="tw_csv_report_bar")

                        with tempfile.TemporaryDirectory() as tmpdir:
                            image_paths = save_two_way_matplotlib_images(df_long, tmpdir)
                            pdf_bytes = generate_pdf_with_images("Two-Way ANOVA Report", report_text, image_paths)
                            st.download_button(
                                "Download PDF Report",
                                pdf_bytes,
                                file_name="two_way_anova_report.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )

                except Exception as e:
                    st.markdown(f'<div class="danger-box">Error: {str(e)}</div>', unsafe_allow_html=True)