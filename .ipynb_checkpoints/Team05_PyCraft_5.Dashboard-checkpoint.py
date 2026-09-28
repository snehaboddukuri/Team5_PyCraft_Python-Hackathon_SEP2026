"""
Team05 PyCraft — Heart Failure Analytics Dashboard
Descriptive · Prescriptive · Predictive

Run:  streamlit run app.py
Data: Team05_PyCraft_Cleaned_data/wide_merged_table.csv  (or upload it in the sidebar)
"""
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (roc_auc_score, roc_curve, average_precision_score, confusion_matrix,
                             accuracy_score, precision_score, recall_score, f1_score, classification_report)

# ----------------------------------------------------------------------------------------------
# Page setup & style
# ----------------------------------------------------------------------------------------------
st.set_page_config(page_title="Hospitalized HF Patients Analytics · Team05 PyCraft", page_icon="🫀", layout="wide")

# Validated categorical palette (fixed order) + reserved status colours
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#6fa3e0", "#f29a6e", "#6cc7a1", "#f2c45a", "#eea3c0", "#7fbf7f", "#9a8fd6", "#e88482")   # soft/medium tones
INK, INK2, GRID, BORDER = "#0b0b0b", "#52514e", "#e6e4de", "#cfccc3"

# ---- One colour per concept, used identically on every chart ----
READM_C = BLUE          # readmission (any horizon)          -> blue family
DEATH_C = RED           # death / mortality (any horizon)     -> red family
GOOD_C = AQUA           # alive / regular discharge / safe
DAMA_C = ORANGE         # left against medical advice
NUTR_C = YELLOW         # nutrition (BMI, albumin)
FEMALE_C, MALE_C = MAGENTA, VIOLET
NEUTRAL = "#4b5d73"     # plain counts / prevalence (medicines, comorbidities)
READM_RAMP = ["#e6f0fb", "#c9def5", "#a4c6ee", "#7eaee5", "#5b91d1", "#4677b4"]   # light -> dark = low -> high risk
DEATH_RAMP = ["#fbe3e2", "#f6c3c2", "#f0a3a1", "#e88482", "#d7625f", "#c04d4a"]
SEV_RAMP = ["#e2def7", "#b3aaea", "#7d6fd6", "#4a3aa7", "#2c2170"]                # severity grades (NYHA / Killip)


def ramp(colors, n):
    """n evenly spaced steps from a light->dark ramp (lowest band lightest, riskiest band darkest)."""
    if n <= 1:
        return [colors[-1]]
    idx = np.linspace(0, len(colors) - 1, n).round().astype(int)
    return [colors[i] for i in idx]


def outcome_color(col):
    return DEATH_C if 'death' in col else READM_C


def outcome_ramp(col, n):
    return ramp(DEATH_RAMP if 'death' in col else READM_RAMP, n)

st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
.kpi {background:#ffffff; border:1px solid #e6e4de; border-radius:12px; padding:14px 16px; height:100%; min-height:132px;}
.kpi .lbl {font-size:0.78rem; color:#52514e; text-transform:uppercase; letter-spacing:.04em;}
.kpi .val {font-size:1.75rem; font-weight:700; color:#0b0b0b; line-height:1.2; margin-top:2px;}
.kpi .sub {font-size:0.78rem; color:#52514e; margin-top:2px;}
.kpi.accent-blue {border-top:4px solid #2a78d6;}
.kpi.accent-red {border-top:4px solid #e34948;}
.kpi.accent-aqua {border-top:4px solid #1baf7a;}
.kpi.accent-violet {border-top:4px solid #4a3aa7;}
.kpi.accent-orange {border-top:4px solid #eb6834;}
.kpi.accent-slate {border-top:4px solid #4b5d73;}
.insight {background:#f4f8fd; border-left:4px solid #2a78d6; border-radius:6px; padding:10px 14px; margin:4px 0 14px 0; font-size:0.92rem;}
.action {background:#f2faf6; border-left:4px solid #1baf7a; border-radius:6px; padding:10px 14px; margin:4px 0 14px 0; font-size:0.92rem;}
.warn {background:#fdf5ec; border-left:4px solid #eb6834; border-radius:6px; padding:10px 14px; margin:4px 0 14px 0; font-size:0.92rem;}
h3 {margin-top: 0.6rem;}
.figcap {text-align:center; font-size:0.86rem; color:#3b3a37; margin:6px 4px 4px 4px; line-height:1.45; border:1px solid #cfccc3; border-radius:8px; background:#f7f6f2; padding:8px 14px;}
.figcap b {color:#0b0b0b;}
.charttitle {text-align:center; font-weight:400; font-size:1.55rem; color:#000000; letter-spacing:0.01em; margin:2px 0 0 0; padding:4px 8px 8px 8px; border-bottom:1px solid #cfccc3;}
</style>
""", unsafe_allow_html=True)

_ST_VER = tuple(int(x) for x in st.__version__.split(".")[:2])
_NEW_WIDTH = _ST_VER >= (1, 46)


_FIG_NO = [0]


def _caption(text):
    st.markdown(f'<div class="figcap">{text}</div>', unsafe_allow_html=True)


def show(fig, caption=""):
    """Plotly chart inside a bordered card, followed by a numbered caption."""
    with st.container(border=True):
        title = fig.layout.title.text
        if title:   # render the title as a centred, bordered HTML label above the chart
            st.markdown(f'<div class="charttitle">{title}</div>', unsafe_allow_html=True)
            legend_on = fig.layout.showlegend is not False and (fig.layout.legend.y is None or fig.layout.legend.y > 0.5)
            is_pie = any(t.type == 'pie' for t in fig.data)
            has_facets = any(getattr(an, 'yref', None) == 'paper' and (an.y or 0) >= 1 for an in fig.layout.annotations)
            fig.update_layout(title_text="", margin=dict(t=45 if (legend_on or is_pie or has_facets) else 15))
        if _NEW_WIDTH:
            st.plotly_chart(fig, width="stretch")
        else:
            st.plotly_chart(fig, use_container_width=True)
        if caption:
            _caption(caption)


def show_mpl(fig, caption="", title=None):
    """Matplotlib figure inside a bordered card, with a centred title + divider and a caption."""
    with st.container(border=True):
        if title:
            st.markdown(f'<div class="charttitle">{title}</div>', unsafe_allow_html=True)
        st.pyplot(fig)
        plt.close(fig)
        if caption:
            _caption(caption)


def table(data, **kw):
    if _NEW_WIDTH:
        st.dataframe(data, width="stretch", **kw)
    else:
        st.dataframe(data, use_container_width=True, **kw)


KPI_COLORS = {"patients": "#7fa7d9", "readm": "#6bb6c9", "death": "#e39a9a", "los": "#b39ddb",
              "severe": "#f0a875", "emergency": "#86c9a6", "dama": "#e2c26a", "echo": "#d8a2c8"}


def kpi(col, label, value, sub="", accent="patients"):
    c = KPI_COLORS.get(accent, "#b8b5ad")
    col.markdown(f'<div class="kpi" style="border:2px solid {c}; border-top:6px solid {c};"><div class="lbl">{label}</div>'
                 f'<div class="val">{value}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def insight(text):
    st.markdown(f'<div class="insight">💡 {text}</div>', unsafe_allow_html=True)


def action(text):
    st.markdown(f'<div class="action">✅ <b>Action:</b> {text}</div>', unsafe_allow_html=True)


def caution(text):
    st.markdown(f'<div class="warn">⚠️ {text}</div>', unsafe_allow_html=True)


def style(fig, height=380, legend=True):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=95 if legend else 70, b=10),
                      plot_bgcolor="white", paper_bgcolor="white", font=dict(color=INK, size=12), showlegend=legend,
                      title=dict(x=0.5, xanchor="center", y=0.97, yanchor="top", font=dict(size=19, color=INK)),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, title=dict(text="")),
                      hoverlabel=dict(bgcolor="white", font_size=12))
    fig.update_xaxes(showgrid=False, zeroline=False, showline=True, linecolor=BORDER, ticks="outside",
                     tickcolor=BORDER, tickfont=dict(color="#222222", size=13), title_font=dict(color="#000000", size=16, family="Arial, Helvetica, sans-serif"))
    fig.update_yaxes(showgrid=False, zeroline=False, showline=True, linecolor=BORDER, ticks="outside",
                     tickcolor=BORDER, tickfont=dict(color="#222222", size=13), title_font=dict(color="#000000", size=16, family="Arial, Helvetica, sans-serif"))
    return fig


from matplotlib.colors import LinearSegmentedColormap
SOFT_BLUES = LinearSegmentedColormap.from_list('soft_blues', ['#f4f8fd', '#c9def5', '#7eaee5', '#4677b4'])


def mpl_style(ax):
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color(BORDER)
    ax.grid(False)
    ax.tick_params(colors='#222222', labelsize=10)


def mpl_title(ax, text):
    ax.set_title(text, fontsize=14, color="#000000", loc="center", pad=12)


AGE_ORDER = ['21-29', '29-39', '39-49', '49-59', '59-69', '69-79', '79-89', '89-110']
AGE_MID = {'21-29': 25, '29-39': 34, '39-49': 44, '49-59': 54, '59-69': 64, '69-79': 74, '79-89': 84, '89-110': 95}


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(file):
    df = pd.read_csv(file, low_memory=False)
    echo_cols = ['heart_pumping_efficiency', 'left_ventricle_size', 'mitral_valve_ems', 'ea_ratio', 'pulmonary_velocity']
    new = {'age_mid': df['age_category'].map(AGE_MID)}
    new['age_group'] = np.where(new['age_mid'] >= 69, 'Elderly (69+)', 'Younger (<69)')
    new['male'] = (df['gender'] == 'Male').astype(int)
    new['emergency'] = (df['admission_type'] == 'Emergency').astype(int)
    new['on_oxygen'] = (df['oxygen_therapy'] == 'OxygenTherapy').astype(int)
    new['conscious_clear'] = (df['consciousness'] == 'Clear').astype(int)
    for t in ['Left', 'Right', 'Both']:
        new['hf_' + t.lower()] = (df['heart_failure_type'] == t).astype(int)
    for w in ['Cardiology', 'GeneralWard', 'ICU', 'Others']:
        new['adm_' + w] = (df['admission_ward'] == w).astype(int)
    new['echo_recorded'] = df[echo_cols].notna().any(axis=1)
    new['underweight'] = df['bmi'] < 18.5
    new['dama'] = (df['hospitalization_outcome'] == 'DischargeAgainstOrder').astype(int)
    df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    return df


DEFAULT_PATH = "Team05_PyCraft_Cleaned_data/wide_merged_table.csv"
with st.sidebar:
    st.markdown("## 🫀 Hospitalized HF Patients Analytics")
    st.caption("Team05 PyCraft · 2,008 heart-failure admissions (Sichuan, 2016–2019)")
    uploaded = st.file_uploader("Data file (wide_merged_table.csv)", type="csv",
                                help=f"Leave empty to use {DEFAULT_PATH}")

if uploaded is not None:
    df_all = load_data(uploaded)
elif os.path.exists(DEFAULT_PATH):
    df_all = load_data(DEFAULT_PATH)
else:
    st.error(f"Data not found. Put **{DEFAULT_PATH}** next to app.py, or upload the CSV in the sidebar.")
    st.stop()

# ---------------------------- Sidebar filters (Descriptive + Prescriptive) ---------------------
with st.sidebar:
    st.markdown("### Filters")
    st.caption("Apply to the KPIs, Descriptive and Prescriptive tabs. Models in the Predictive tab always use all patients.")
    f_gender = st.multiselect("Gender", sorted(df_all['gender'].unique()), default=sorted(df_all['gender'].unique()))
    f_age = st.multiselect("Age group", AGE_ORDER, default=AGE_ORDER)
    f_type = st.multiselect("Heart-failure type", ['Both', 'Left', 'Right'], default=['Both', 'Left', 'Right'])
    f_adm = st.multiselect("Admission type", ['Emergency', 'NonEmergency'], default=['Emergency', 'NonEmergency'])
    f_sev = st.select_slider("NYHA class (heart_failure_severity)", options=[2, 3, 4], value=(2, 4))
    st.markdown("---")
    st.caption("Built with Streamlit · Plotly · Matplotlib · scikit-learn")

df = df_all[df_all['gender'].isin(f_gender) & df_all['age_category'].isin(f_age) &
            df_all['heart_failure_type'].isin(f_type) & df_all['admission_type'].isin(f_adm) &
            df_all['heart_failure_severity'].between(f_sev[0], f_sev[1])].copy()

if len(df) < 30:
    st.warning(f"Only {len(df)} patients match the filters — widen them to see reliable charts.")
    if len(df) == 0:
        st.stop()

# ----------------------------------------------------------------------------------------------
# Header + KPI row
# ----------------------------------------------------------------------------------------------
st.markdown("<h1 style='text-align:center; margin-bottom:0.2rem;'>🫀 Hospitalized Heart Failure Patients Analytics Dashboard</h1>", unsafe_allow_html=True)
st.divider()

pct = lambda s: f"{s.mean() * 100:.1f}%"
r1 = st.columns(4)
kpi(r1[0], "Patients", f"{len(df):,}", f"{(df['gender'] == 'Female').mean()*100:.0f}% female · "
                                        f"{(df['age_mid'] >= 69).mean()*100:.0f}% aged 69+", "patients")
kpi(r1[1], "6-month readmission", pct(df['readmission_6m']),
    f"28-day: {pct(df['readmission_28d'])} · 3-month: {pct(df['readmission_3m'])}", "readm")
kpi(r1[2], "6-month mortality", pct(df['death_6m']),
    f"28-day: {pct(df['death_28d'])} · in-hospital: {(df['hospitalization_outcome'] == 'Dead').sum()} deaths", "death")
kpi(r1[3], "Median length of stay", f"{df['discharge_day'].median():.0f} days",
    f"IQR {df['discharge_day'].quantile(.25):.0f}–{df['discharge_day'].quantile(.75):.0f} days", "los")
st.write("")
r2 = st.columns(4)
kpi(r2[0], "Severe heart failure", pct(df['severe_hf']),
    f"NYHA class 4: {(df['heart_failure_severity'] == 4).mean()*100:.0f}% · Killip 3–4: "
    f"{(df['heart_damage_level'] >= 3).mean()*100:.0f}%", "severe")
kpi(r2[1], "Emergency admissions", pct(df['emergency']), f"ICU admissions: {(df['admission_ward'] == 'ICU').sum()}", "emergency")
kpi(r2[2], "Left against medical advice", pct(df['dama']),
    f"{df['dama'].sum()} patients · 28-day death {df.loc[df['dama'] == 1, 'death_28d'].mean()*100:.0f}%"
    if df['dama'].sum() else "0 patients", "dama")
kpi(r2[3], "Echo gap", f"{(~df['echo_recorded']).mean()*100:.1f}%",
    f"patients with no echo recorded · mean CCI {df['cci_score'].mean():.1f}", "echo")
st.write("")

tab_d, tab_p, tab_m = st.tabs(["📊  Descriptive — what happened?", "🩺  Prescriptive — what should we do?",
                               "🤖  Predictive — what will happen?"])

# ==============================================================================================
# TAB 1 — DESCRIPTIVE
# ==============================================================================================
with tab_d:
    st.markdown("### Who are the patients?")
    c1, c2 = st.columns(2)
    with c1:
        ag = df.groupby(['age_category', 'gender']).size().reset_index(name='Patients')
        fig = px.bar(ag, x='age_category', y='Patients', color='gender', barmode='group',
                     category_orders={'age_category': AGE_ORDER}, color_discrete_map={'Female': MAGENTA, 'Male': BLUE},
                     title="Age & gender distribution", text='Patients')
        fig.update_traces(textposition='outside', marker_line_width=0, hovertemplate="%{x}: %{y} patients")
        fig.update_layout(xaxis_title="Age group", yaxis_title="Patients", bargap=0.25, bargroupgap=0.08)
        show(style(fig), "Number of patients in each age group, split by gender (pink = female, blue = male). "
                         "The cohort is concentrated in the 69–89 age groups, where women outnumber men.")
    with c2:
        uw = (df.groupby('age_category')['underweight'].mean() * 100).reindex(AGE_ORDER).dropna().reset_index()
        uw.columns = ['Age group', 'Underweight %']
        fig = px.bar(uw, x='Age group', y='Underweight %', title="Malnutrition risk: % underweight (BMI < 18.5) by age",
                     text=uw['Underweight %'].round(1).astype(str) + '%')
        fig.update_traces(marker_color=ORANGE, textposition='outside', hovertemplate="%{x}: %{y:.1f}% underweight")
        fig.add_hline(y=df['underweight'].mean() * 100, line_dash="dot", line_color=INK2,
                      annotation_text=f"overall {df['underweight'].mean()*100:.1f}%", annotation_position="bottom left")
        fig.update_layout(yaxis_title="% underweight", title="Malnutrition risk by age")
        show(style(fig, legend=False), "Share of patients who are underweight (BMI < 18.5) in each age group; the dotted line is "
                                       "the cohort average. Underweight rises with age and is highest in the oldest patients.")
    oldest = df.loc[df['age_category'] == '89-110', 'underweight']
    oldest_txt = f", rising to <b>{oldest.mean()*100:.0f}%</b> in patients aged 89+" if len(oldest) else ""
    insight(f"The cohort is elderly (<b>{(df['age_mid'] >= 69).mean()*100:.0f}%</b> aged 69+) and malnutrition is common: "
            f"<b>{df['underweight'].mean()*100:.0f}%</b> are underweight{oldest_txt}.")

    st.markdown("### When do readmission and death happen?")
    c1, c2 = st.columns([3, 2])
    with c1:
        hz = pd.DataFrame({'Horizon': ['28 days', '3 months', '6 months'] * 2,
                           'Rate %': [df['readmission_28d'].mean()*100, df['readmission_3m'].mean()*100, df['readmission_6m'].mean()*100,
                                      df['death_28d'].mean()*100, df['death_3m'].mean()*100, df['death_6m'].mean()*100],
                           'Outcome': ['Readmission'] * 3 + ['Death'] * 3})
        fig = px.line(hz, x='Horizon', y='Rate %', color='Outcome', markers=True, text=hz['Rate %'].round(1).astype(str) + '%',
                      color_discrete_map={'Readmission': READM_C, 'Death': DEATH_C},
                      title="Cumulative readmission and mortality after admission")
        fig.update_traces(line_width=2.5, marker_size=9, textposition='top center', hovertemplate="%{x}: %{y:.1f}%")
        fig.update_layout(yaxis_title="% of patients", yaxis_range=[0, max(hz['Rate %']) * 1.25],
                          title="Readmission and mortality over time")
        show(style(fig), "Cumulative % of patients readmitted (blue) or dead (red) by 28 days, 3 months and 6 months. "
                         "Readmission keeps climbing for 6 months; mortality is low and almost flat after the first month.")
    with c2:
        oc = df['hospitalization_outcome'].value_counts().reset_index()
        oc.columns = ['Outcome', 'Patients']
        fig = px.pie(oc, names='Outcome', values='Patients', hole=0.6, title="How the hospital stay ended",
                     color='Outcome', color_discrete_map={'Alive': GOOD_C, 'DischargeAgainstOrder': DAMA_C, 'Dead': DEATH_C})
        fig.update_traces(textinfo='percent', textposition='outside', hovertemplate="%{label}: %{value} patients (%{percent})",
                          marker=dict(line=dict(color='white', width=2)))
        fig = style(fig); fig.update_layout(margin=dict(t=40, b=40, l=30, r=30), legend=dict(yanchor='top', y=-0.05, x=0.5, xanchor='center'))
        show(fig, "How each hospital stay ended: discharged alive (aqua), left against medical advice (orange) or died in "
                  "hospital (red). In-hospital death is rare; about 1 in 20 patients leave against advice.")
    insight("Readmission, not death, is the main burden: it keeps rising through 6 months, while almost all deaths happen "
            "in the first 4 weeks. Mortality prevention must be front-loaded; readmission prevention must last 6 months.")

    st.markdown("### How sick are patients on arrival?")
    c1, c2 = st.columns(2)
    with c1:
        sev_scale = st.radio("Severity scale", ["NYHA class (heart_failure_severity)", "Killip class (heart_damage_level)"],
                             horizontal=True, key="sevscale")
        col = 'heart_failure_severity' if sev_scale.startswith("NYHA") else 'heart_damage_level'
        sv = (pd.crosstab(df['gender'], df[col], normalize='index') * 100).reset_index().melt(id_vars='gender', var_name='Level', value_name='%')
        sv['Level'] = sv['Level'].astype(str)
        sev_map = {'1': '#c9def5', '2': '#a4c6ee', '3': '#7eaee5', '4': '#4677b4'} if col == 'heart_damage_level' \
            else {'2': '#c9def5', '3': '#7eaee5', '4': '#4677b4'}
        fig = px.bar(sv, y='gender', x='%', color='Level', orientation='h', color_discrete_map=sev_map,
                     title=f"{sev_scale.split(' (')[0]} at admission by gender",
                     text=sv['%'].round(0).astype(int).astype(str) + '%')
        fig.update_traces(marker_line_color='white', marker_line_width=2, hovertemplate="Level %{fullData.name}: %{x:.1f}%")
        fig.update_layout(xaxis_title="% of patients within each gender", yaxis_title="", legend_title_text="Level")
        show(style(fig, 340), "Each bar is 100% of one gender, split by severity level (light blue = mildest, dark blue = "
                              "most severe). Men have a slightly larger share in the most severe level.")
    with c2:
        t = df.groupby('heart_failure_type').agg(Patients=('patient_id', 'size'), Death=('death_6m', 'mean'),
                                                 Readmission=('readmission_6m', 'mean')).reset_index()
        tl = t.melt(id_vars=['heart_failure_type', 'Patients'], value_vars=['Readmission', 'Death'], var_name='Outcome', value_name='Rate')
        tl['Rate'] *= 100
        fig = px.bar(tl, x='heart_failure_type', y='Rate', color='Outcome', barmode='group', facet_col='Outcome',
                     color_discrete_map={'Readmission': READM_C, 'Death': DEATH_C}, text=tl['Rate'].round(1).astype(str) + '%',
                     title="6-month outcomes by HF type", hover_data={'Patients': True},
                     category_orders={'heart_failure_type': ['Left', 'Right', 'Both']})
        fig.update_yaxes(matches=None, title_text="")
        fig.update_yaxes(title_text='% of patients', row=1, col=1)
        fig.update_traces(textposition='outside')
        fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1], font=dict(size=15, color='#000000')))
        fig.update_layout(xaxis_title="", xaxis2_title="")
        show(style(fig, 340, legend=False), "6-month readmission (blue, left) and death (red, right) for left-sided, "
                                            "right-sided and biventricular (Both) heart failure. Biventricular failure has the "
                                            "highest rates; right-sided has only 51 patients.")
    insight("Men arrive slightly sicker (more NYHA 4 and Killip 4 shock). Biventricular (\"Both\") failure is the most common "
            "and the riskiest type; right-sided failure has too few patients (51) to judge.")

    st.markdown("### Consciousness at admission and early death")
    cons_order = ['Clear', 'ResponsiveToSound', 'ResponsiveToPain', 'Nonresponsive']
    cons_col = {'Clear': BLUE, 'ResponsiveToSound': YELLOW, 'ResponsiveToPain': ORANGE, 'Nonresponsive': RED}
    fig = go.Figure()
    for lvl in cons_order:
        g = df[df['consciousness'] == lvl]
        if len(g) == 0:
            continue
        days = np.arange(0, 29)
        cum = [(g['death_time_days'] <= d).mean() * 100 for d in days]
        fig.add_trace(go.Scatter(x=days, y=cum, mode='lines', line_shape='hv', line=dict(color=cons_col[lvl], width=3.5),
                                 name=f"{lvl} (n={len(g)})", hovertemplate=f"{lvl}<br>Day %{{x}}: %{{y:.1f}}% died<extra></extra>"))
    fig.update_layout(title="Early mortality by consciousness at admission",
                      xaxis_title="Days since admission", yaxis_title="% of group who died", hovermode="x unified")
    fig.update_xaxes(tickvals=[0, 7, 14, 21, 28])
    show(style(fig, 400), "Cumulative % of each consciousness group who died, day by day, over the first 28 days (blue = "
                          "clear; yellow, orange and red = increasingly impaired). The more impaired the consciousness, the higher and earlier "
                          "the deaths.")
    insight("Only ~2% of patients arrive with reduced consciousness, but they account for over a third of early deaths — "
            "and most of those deaths happen in week 1.")

    st.markdown("### Treatment & comorbidity landscape")
    c1, c2 = st.columns(2)
    with c1:
        drug_cols = [c for c in df.columns if c.startswith('drug_')]
        dr = (df[drug_cols].mean() * 100).sort_values(ascending=True).tail(12)
        dr.index = [i.replace('drug_', '').replace('_', ' ') for i in dr.index]
        fig = px.bar(x=dr.values, y=dr.index, orientation='h', title="Most-used medicines",
                     text=[f"{v:.0f}%" for v in dr.values])
        fig.update_traces(marker_color=VIOLET, textposition='outside', hovertemplate="%{y}: %{x:.1f}% of patients")
        fig.update_layout(xaxis_title="% of patients", yaxis_title="", xaxis_range=[0, 110])
        show(style(fig, 440, legend=False), "The 12 medicines given to the most patients during the stay. Diuretics "
                                            "(spironolactone, furosemide) dominate, followed by digitalis and inotropes.")
    with c2:
        como = {'diabetes': 'Diabetes', 'moderate_to_severe_chronic_kidney_disease': 'Chronic kidney disease',
                'chronic_obstructive_pulmonary_disease': 'COPD', 'cerebrovascular_disease': 'Stroke', 'dementia': 'Dementia',
                'type2_respiratory_failure': 'Type 2 resp. failure', 'liver_disease': 'Liver disease', 'solid_tumor': 'Solid tumour',
                'peptic_ulcer_disease': 'Peptic ulcer', 'connective_tissue_disease': 'Connective tissue dis.'}
        cp = pd.Series({v: df[k].mean() * 100 for k, v in como.items()}).sort_values()
        fig = px.bar(x=cp.values, y=cp.index, orientation='h', title="Comorbidity prevalence",
                     text=[f"{v:.1f}%" for v in cp.values])
        fig.update_traces(marker_color=AQUA, textposition='outside', hovertemplate="%{y}: %{x:.1f}%")
        fig.update_layout(xaxis_title="% of patients", yaxis_title="", xaxis_range=[0, cp.max() * 1.25])
        show(style(fig, 440, legend=False), "Share of patients with each long-term condition. Chronic kidney disease and "
                                            "diabetes are the most common, each affecting about 1 in 4 patients.")
    insight("Care focuses on decongestion (spironolactone, furosemide) and rate control/inotropy (digoxin, deslanoside, milrinone). "
            "Diabetes and chronic kidney disease are the most common comorbidities (~1 in 4 patients each).")

    st.markdown("### Explore any variable")
    num_cols = ['bmi', 'pulse', 'sbp', 'dbp', 'creatinine', 'glomerular_filtration_rate', 'sodium', 'potassium', 'albumin',
                'hemoglobin', 'brain_natriuretic_peptide', 'high_sensitivity_troponin', 'uric_acid', 'discharge_day', 'cci_score']
    c1, c2 = st.columns([1, 3])
    with c1:
        var = st.selectbox("Variable", num_cols, index=4)
        split = st.selectbox("Compare by", ['death_6m', 'readmission_6m', 'readmission_28d', 'gender', 'heart_failure_type', 'age_group'])
        clip = st.checkbox("Hide extreme values (top/bottom 1%)", value=True)
    with c2:
        d = df[[var, split]].dropna().copy()
        if clip and len(d) > 20:
            lo, hi = d[var].quantile([0.01, 0.99])
            d = d[d[var].between(lo, hi)]
        d[split] = d[split].astype(str).replace({'0': 'No', '1': 'Yes'})
        fig = px.box(d, x=split, y=var, color=split, points=False, title=f"{var} by {split}",
                     color_discrete_sequence=[BLUE, ORANGE, AQUA, YELLOW])
        fig.update_layout(xaxis_title=split, yaxis_title=var)
        show(style(fig, 380, legend=False), f"Distribution of <b>{var}</b> in each <b>{split}</b> group. The line in each box "
                                            "is the median, the box holds the middle 50% of patients and the whiskers the typical range.")
        med = d.groupby(split)[var].median().round(1).to_dict()
        st.caption("Medians: " + " · ".join(f"{k}: {v}" for k, v in med.items()))

# ==============================================================================================
# TAB 2 — PRESCRIPTIVE
# ==============================================================================================
with tab_p:
    st.markdown("Each panel turns an analysis finding into an action. All charts respond to the sidebar filters.")

    # ---- 1. Kidney stage
    st.markdown("### 1 · Kidney function: two different risks at two different stages")
    def gfr_stage(g):
        if pd.isna(g): return None
        return ('G1 (≥90)' if g >= 90 else 'G2 (60-89)' if g >= 60 else 'G3a (45-59)' if g >= 45 else
                'G3b (30-44)' if g >= 30 else 'G4 (15-29)' if g >= 15 else 'G5 (<15)')
    stages = ['G1 (≥90)', 'G2 (60-89)', 'G3a (45-59)', 'G3b (30-44)', 'G4 (15-29)', 'G5 (<15)']
    df = df.assign(gfr_stage=df['glomerular_filtration_rate'].apply(gfr_stage))
    gs = df.groupby('gfr_stage').agg(n=('patient_id', 'size'), Readmission=('readmission_6m', 'mean'),
                                     Death=('death_6m', 'mean')).reindex(stages).dropna(how='all').reset_index()
    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(gs, x='gfr_stage', y=gs['Readmission'] * 100, markers=True, title="6-month readmission by eGFR stage",
                      text=(gs['Readmission'] * 100).round(1).astype(str) + '%', hover_data={'n': True})
        fig.update_traces(line_color=READM_C, line_width=2.5, marker_size=9, textposition='top center')
        fig.update_layout(xaxis_title="Kidney stage (best → worst)", yaxis_title="% readmitted", yaxis_range=[0, 65])
        show(style(fig, 340, legend=False), "6-month readmission for each kidney-function (eGFR) stage, from normal (G1) to "
                                            "kidney failure (G5). Readmission peaks at moderate disease (G3b) and then falls.")
    with c2:
        fig = px.line(gs, x='gfr_stage', y=gs['Death'] * 100, markers=True, title="6-month mortality by eGFR stage",
                      text=(gs['Death'] * 100).round(1).astype(str) + '%', hover_data={'n': True})
        fig.update_traces(line_color=DEATH_C, line_width=2.5, marker_size=9, textposition='top center')
        fig.update_layout(xaxis_title="Kidney stage (best → worst)", yaxis_title="% died", yaxis_range=[0, 13])
        show(style(fig, 340, legend=False), "6-month mortality for each eGFR stage. Death stays low until G3b and then jumps "
                                            "sharply at severe kidney failure (G4–G5).")
    insight("Readmission peaks at moderate kidney disease (G3b), while mortality jumps only at G4–G5 — a straight-line eGFR score misses both.")
    action("eGFR 30–59 → readmission-prevention pathway (7–14-day clinic, repeat kidney tests, diuretic review). "
           "eGFR < 30 → nephrology co-management, renal dosing and early goals-of-care discussion.")

    # ---- 2. Severity x admission type heatmap (matplotlib)
    st.markdown("### 2 · Who needs the closest follow-up?")
    c1, c2 = st.columns([3, 2])
    with c1:
        grid = pd.crosstab(df['heart_failure_severity'], df['admission_type'], values=df['readmission_28d'], aggfunc='mean') * 100
        cnt = pd.crosstab(df['heart_failure_severity'], df['admission_type'])
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        im = ax.imshow(grid.values, cmap=SOFT_BLUES, vmin=0, vmax=max(12, np.nanmax(grid.values)), aspect='auto')
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                v = grid.values[i, j]
                ax.text(j, i, f"{v:.1f}%\n(n={cnt.values[i, j]})", ha='center', va='center', fontsize=10, fontweight='bold',
                        color='white' if v > 9.5 else INK)
        ax.set_xticks(range(grid.shape[1])); ax.set_xticklabels(grid.columns)
        ax.set_yticks(range(grid.shape[0])); ax.set_yticklabels([f"NYHA {s}" for s in grid.index])
        ax.set_xlabel("Admission type", fontsize=12, color='#000000')
        ax.set_ylabel("NYHA class", fontsize=12, color='#000000')
        cb = plt.colorbar(im, ax=ax, fraction=0.04); cb.set_label("% readmitted", color="#000000", fontsize=12); cb.outline.set_edgecolor(BORDER)
        mpl_style(ax); ax.tick_params(length=0); plt.tight_layout()
        show_mpl(fig, title="28-day readmission by severity", caption="28-day readmission rate for each combination of NYHA class (rows) and admission type (columns); darker "
                      "blue = higher readmission, n = patients in the cell. Risk rises with severity and is highest for "
                      "emergency NYHA 4 admissions.")
    with c2:
        hr = (df['admission_type'] == 'Emergency') & (df['heart_failure_severity'] == 4)
        st.metric("High-risk profile (Emergency + NYHA 4)", f"{hr.sum()} patients",
                  f"{hr.mean()*100:.0f}% of cohort", delta_color="off")
        st.metric("28-day readmission — high-risk vs rest",
                  f"{df.loc[hr, 'readmission_28d'].mean()*100:.1f}%" if hr.any() else "–",
                  f"{(df.loc[hr, 'readmission_28d'].mean() - df.loc[~hr, 'readmission_28d'].mean())*100:+.1f} pts vs rest" if hr.any() and (~hr).any() else None,
                  delta_color="inverse")
        st.metric("28-day mortality — high-risk vs rest",
                  f"{df.loc[hr, 'death_28d'].mean()*100:.1f}%" if hr.any() else "–",
                  f"{(df.loc[hr, 'death_28d'].mean() - df.loc[~hr, 'death_28d'].mean())*100:+.1f} pts vs rest" if hr.any() and (~hr).any() else None,
                  delta_color="inverse")
    action("Tier 1 (Emergency + NYHA 4): call within 72 h and clinic within 7 days. Tier 2 (NYHA 4 planned or Emergency + NYHA 3): "
           "clinic within 14 days. Do not tier by department — ward makes little difference.")

    # ---- 3. Cardio-renal & potassium
    st.markdown("### 3 · Biomarker danger zones")
    c1, c2 = st.columns(2)
    with c1:
        both = df[df['heart_failure_type'] == 'Both'].copy()
        if len(both):
            ren = (both['creatinine'] > 133) | (both['glomerular_filtration_rate'] < 60)
            both['group'] = np.select([(both['severe_hf'] == 1) & ren, both['severe_hf'] == 1, ren],
                                      ['Cardiac + renal', 'Cardiac only', 'Renal only'], 'Neither')
            order = ['Neither', 'Renal only', 'Cardiac only', 'Cardiac + renal']
            cr = both.groupby('group').agg(n=('patient_id', 'size'), Death=('death_6m', 'mean')).reindex(order).dropna().reset_index()
            fig = px.bar(cr, x='group', y=cr['Death'] * 100, text=(cr['Death'] * 100).round(1).astype(str) + '%',
                         title="Cardio-renal risk (biventricular HF)", hover_data={'n': True})
            fig.update_traces(marker_color=[READM_RAMP[1], READM_RAMP[2], READM_RAMP[3], RED][:len(cr)], textposition='outside')
            fig.update_layout(xaxis_title="", yaxis_title="% died within 6 months")
            show(style(fig, 360, legend=False), "6-month mortality in biventricular heart failure, grouped by severe HF (cardiac) "
                                                "and abnormal kidney function (renal). The red bar marks patients with both "
                                                "problems, who die far more often.")
    with c2:
        kb = pd.cut(df['potassium'], [0, 3.5, 5.0001, 5.5001, 6.0001, 20], right=False,
                    labels=['<3.5', '3.5–5.0', '5.1–5.5', '5.6–6.0', '>6.0'])
        kt = df.groupby(kb, observed=True).agg(n=('patient_id', 'size'), Death=('death_6m', 'mean'),
                                              Spiro=('drug_Spironolactone_Tablet', 'mean')).reset_index()
        kt.columns = ['Potassium (mmol/L)', 'n', 'Death', 'Spiro']
        fig = px.bar(kt, x='Potassium (mmol/L)', y=kt['Death'] * 100, text=(kt['Death'] * 100).round(1).astype(str) + '%',
                     title="Mortality by potassium", hover_data={'n': True})
        fig.update_traces(marker_color=[READM_RAMP[1], READM_RAMP[2], READM_RAMP[3], READM_RAMP[4], RED][:len(kt)], textposition='outside')
        fig.update_layout(yaxis_title="% died within 6 months")
        show(style(fig, 360, legend=False), "6-month mortality for each band of admission potassium (mmol/L), from low to "
                                            "severely high (red = most severe band). Risk climbs steadily above 5.0.")
        st.caption("Spironolactone use by band: " + " · ".join(f"{r['Potassium (mmol/L)']}: {r['Spiro']*100:.0f}%" for _, r in kt.iterrows()))
    action("Severe HF + abnormal kidneys → automatic nephrology consult. Potassium > 5.0 → flag and recheck in 48–72 h; "
           "≥ 5.5 → treat and reduce/hold spironolactone.")

    # ---- 4. Admission warning signs (interactive)
    st.markdown("### 4 · Admission warning signs — pick a marker")
    signs = {
        "Sodium (hyponatraemia)": ('sodium', [0, 130, 135, 145, 250], ['<130', '130–134', '135–144', '≥145'], 'readmission_28d'),
        "Systolic blood pressure": ('sbp', [0, 100, 140, 400], ['<100', '100–139', '≥140'], 'death_6m'),
        "Albumin (nutrition)": ('albumin', [0, 30, 35, 100], ['<30', '30–34', '≥35'], 'death_6m'),
        "Heart rate": ('pulse', [0, 60, 101, 400], ['<60', '60–100', '>100'], 'death_28d'),
        "GCS (consciousness)": ('gcs', [0, 9, 13, 15, 16], ['3–8', '9–12', '13–14', '15'], 'death_6m'),
    }
    c1, c2 = st.columns([1, 3])
    with c1:
        sname = st.radio("Marker", list(signs), key="sign")
        col, bins, labels, default_out = signs[sname]
        outs = ['death_28d', 'death_6m', 'readmission_28d', 'readmission_6m']
        out = st.selectbox("Outcome", outs, index=outs.index(default_out))
    with c2:
        b = pd.cut(df[col], bins, right=False, labels=labels)
        tt = df.groupby(b, observed=True).agg(n=('patient_id', 'size'), rate=(out, 'mean')).reset_index()
        overall = df[out].mean() * 100
        fig = px.bar(tt, x=col, y=tt['rate'] * 100, text=[f"{r*100:.1f}%<br>n={n}" for r, n in zip(tt['rate'], tt['n'])],
                     title=f"{out} by {sname.split(' (')[0].lower()}")
        colr = DEATH_C if out.startswith('death') else READM_C
        fig.update_traces(marker_color=colr, textposition='outside',
                          hovertemplate="%{x}: %{y:.1f}%<extra></extra>")
        fig.add_hline(y=overall, line_dash='dot', line_color=INK2, annotation_text=f"cohort {overall:.1f}%")
        fig.update_layout(xaxis_title=sname, yaxis_title="% of group", yaxis_range=[0, max(tt['rate'].max() * 100 * 1.35, 1)])
        show(style(fig, 360, legend=False), f"Rate of <b>{out}</b> in each band of <b>{sname.lower()}</b> at admission "
                                            "(n = patients per band); the dotted line is the cohort average. Red = death outcomes, "
                                            "blue = readmission outcomes.")
    action("Low sodium, SBP < 100, albumin < 35 or any GCS < 15 on admission → senior review, closer monitoring and follow-up within 7 days.")

    # ---- 5. Comorbidity risk ratios
    st.markdown("### 5 · Which comorbidities raise which risk?")
    rows = []
    for k, v in como.items():
        has, no = df[df[k] == 1], df[df[k] == 0]
        if len(has) >= 10 and len(no) and no['death_6m'].mean() > 0:
            rows.append({'Condition': v, 'n': len(has), '6-month death': has['death_6m'].mean() / no['death_6m'].mean(),
                         '6-month readmission': has['readmission_6m'].mean() / no['readmission_6m'].mean()})
    if rows:
        rr = pd.DataFrame(rows).sort_values('6-month death')
        rl = rr.melt(id_vars=['Condition', 'n'], var_name='Outcome', value_name='Times higher')
        fig = px.bar(rl, y='Condition', x='Times higher', color='Outcome', barmode='group', orientation='h',
                     color_discrete_map={'6-month death': DEATH_C, '6-month readmission': READM_C}, hover_data={'n': True},
                     title="Comorbidity risk ratios")
        fig.add_vline(x=1, line_dash='dash', line_color=INK2)
        fig.update_traces(hovertemplate="%{y}: %{x:.2f}× higher<extra></extra>")
        fig.update_layout(yaxis_title="", xaxis_title="Times higher than patients without the condition")
        show(style(fig, 460), "How many times more likely 6-month death (red) and readmission (blue) are for patients with "
                              "each condition than for those without it; the dashed line at 1 means no difference. Liver disease, "
                              "type 2 respiratory failure and CKD raise death the most.")
    action("Liver disease, type 2 respiratory failure and CKD → mortality focus (closer monitoring, goals of care). "
           "Dementia, diabetes, CKD → readmission focus (carer involvement, medication support).")

    # ---- 6. Care-process gaps
    st.markdown("### 6 · Care-process gaps")
    c1, c2 = st.columns(2)
    with c1:
        eg = df.groupby(df['echo_recorded'].map({True: 'Echo recorded', False: 'No echo'})).agg(
            Readmission=('readmission_6m', 'mean'), Death=('death_6m', 'mean')).reset_index()
        el = eg.melt(id_vars='echo_recorded', var_name='Outcome', value_name='Rate')
        el['Rate'] *= 100
        fig = px.bar(el, x='echo_recorded', y='Rate', color='Outcome', barmode='group', text=el['Rate'].round(1).astype(str) + '%',
                     color_discrete_map={'Readmission': READM_C, 'Death': DEATH_C}, title="The echo gap")
        fig.update_traces(textposition='outside')
        fig.update_layout(xaxis_title="", yaxis_title="% of patients (6 months)")
        show(style(fig, 360), "6-month readmission (blue) and death (red) for patients with and without an echocardiogram "
                              "recorded. Patients with no echo are readmitted far more often.")
    with c2:
        dm = df[df['hospitalization_outcome'].isin(['Alive', 'DischargeAgainstOrder'])]
        dt = dm.groupby('hospitalization_outcome').agg(**{'28-day death': ('death_28d', 'mean'), 'Severe HF': ('severe_hf', 'mean'),
                                                          '6-month readmission': ('readmission_6m', 'mean')}).reset_index()
        dl = dt.melt(id_vars='hospitalization_outcome', var_name='Measure', value_name='Rate')
        dl['Rate'] *= 100
        dl['hospitalization_outcome'] = dl['hospitalization_outcome'].map({'Alive': 'Regular discharge', 'DischargeAgainstOrder': 'Left against advice'})
        fig = px.bar(dl, x='Measure', y='Rate', color='hospitalization_outcome', barmode='group',
                     text=dl['Rate'].round(1).astype(str) + '%', title="Leaving against medical advice",
                     color_discrete_map={'Regular discharge': GOOD_C, 'Left against advice': DAMA_C})
        fig.update_traces(textposition='outside')
        fig.update_layout(xaxis_title="", yaxis_title="% of patients")
        show(style(fig, 360), "Patients who left against medical advice (orange) compared with regular discharges (aqua). "
                              "They were twice as often severely ill and had a far higher 28-day death rate.")
    action("Make a completed echo a mandatory pre-discharge item. When a patient asks to leave, hold a senior-doctor conversation, "
           "address cost/family reasons, and arrange a 48-hour follow-up call if they still leave.")

# ==============================================================================================
# TAB 3 — PREDICTIVE
# ==============================================================================================
DEMO = ['age_mid', 'male', 'bmi', 'weight', 'height']
COMOR = ['cerebrovascular_disease', 'dementia', 'chronic_obstructive_pulmonary_disease', 'connective_tissue_disease',
         'peptic_ulcer_disease', 'diabetes', 'moderate_to_severe_chronic_kidney_disease', 'hemiplegia', 'malignant_lymphoma',
         'solid_tumor', 'liver_disease', 'aids', 'cci_score', 'type2_respiratory_failure', 'acute_renal_failure',
         'heart_attack_history', 'congestive_heart_failure', 'peripheral_vascular_disease']
CARD = ['heart_failure_severity', 'heart_damage_level', 'severe_hf', 'hf_left', 'hf_right', 'hf_both',
        'heart_pumping_efficiency', 'left_ventricle_size', 'mitral_valve_ems', 'pulmonary_velocity']
VITALS = ['temp', 'pulse', 'resp', 'sbp', 'dbp', 'MAP', 'FiO2', 'on_oxygen']
GCS = ['gcs', 'eye_opening', 'verbal_response', 'movement', 'conscious_clear']
ADMIT = ['emergency', 'adm_Cardiology', 'adm_GeneralWard', 'adm_ICU', 'adm_Others']
SEVERITY = ['heart_failure_severity', 'heart_damage_level', 'severe_hf']


def lab_cols(d):
    cols = list(d.columns)
    return [c for c in cols[cols.index('creatinine'):cols.index('total_hemoglobin') + 1]
            if d[c].isna().mean() < 0.6 and d[c].nunique() > 1]


def rforest():
    return make_pipeline(SimpleImputer(strategy='median'),
                         RandomForestClassifier(n_estimators=200, min_samples_leaf=5, class_weight='balanced_subsample',
                                                random_state=42, n_jobs=-1))


def logreg():
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                         LogisticRegression(max_iter=3000, C=0.1, class_weight='balanced'))


def model_spec(key, d):
    LABS = lab_cols(d)
    no_echo_card = ['heart_failure_severity', 'heart_damage_level', 'severe_hf', 'hf_left', 'hf_right', 'hf_both']
    full = DEMO + COMOR + CARD + VITALS + GCS + LABS + ADMIT
    specs = {
        "28-day death (full admission data)": (d, d['death_28d'], full, 'rf', 0.10),
        "3-month death (full admission data)": (d, d['death_3m'], full, 'rf', 0.10),
        "6-month death (full admission data)": (d, d['death_6m'], full, 'rf', 0.10),
        "28-day death (severity + GCS, simple)": (d, d['death_28d'], SEVERITY + GCS, 'lr', 0.10),
        "HFrEF before echo (10 bedside features)": (None, None, None, 'lr', 0.20),
        "Need for IV inotropes": (d, (d[['drug_Milrinone_Injection', 'drug_Dobutamine_Hydrochloride_Injection',
                                          'drug_Isoprenaline_Hydrochloride_Injection']].sum(axis=1) > 0).astype(int),
                                  full, 'rf', 0.30),
        "Need for digitalis (digoxin/deslanoside)": (d, ((d['drug_Digoxin_Tablet'] + d['drug_Deslanoside_Injection']) > 0).astype(int),
                                                     full, 'lr', 0.50),
    }
    if key == "HFrEF before echo (10 bedside features)":
        e = d[d['heart_pumping_efficiency'].notna()]
        small = ['brain_natriuretic_peptide', 'high_sensitivity_troponin', 'sbp', 'pulse', 'male'] + \
                ['heart_failure_severity', 'heart_damage_level', 'hf_left', 'hf_right', 'hf_both']
        return e, (e['heart_pumping_efficiency'] < 40).astype(int), small, 'lr', 0.20
    return specs[key]


@st.cache_data(show_spinner=False)
def run_model(key, _d, data_id):
    d, y, feats, kind, top = model_spec(key, _d)
    X = d[feats]
    mdl = rforest() if kind == 'rf' else logreg()
    p = cross_val_predict(mdl, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method='predict_proba')[:, 1]
    fitted = mdl.fit(X, y)
    if kind == 'rf':
        imp = pd.Series(fitted[-1].feature_importances_, index=feats)
    else:
        imp = pd.Series(fitted[-1].coef_[0], index=feats)
    return y.values, p, imp, kind, top


with tab_m:
    st.markdown("### Model leaderboard")
    st.caption("Cross-validated results from the Team05 predictive notebook (each patient scored by a model that never saw them).")
    lb = pd.DataFrame([
        ["Death within 28 days", "Random Forest", 0.902, 0.124, 0.676, 0.210],
        ["Death within 3 months", "Random Forest", 0.877, 0.139, 0.667, 0.230],
        ["28-day death: severity + GCS", "Logistic Regression", 0.843, 0.119, 0.649, 0.201],
        ["Elderly (69+) 6-month death", "Random Forest", 0.816, 0.152, 0.571, 0.240],
        ["6-month death: all categories", "Random Forest", 0.798, 0.144, 0.509, 0.225],
        ["6-month death: full admission", "Random Forest", 0.777, 0.144, 0.509, 0.225],
        ["6-month death: + HF subtype", "Random Forest", 0.778, 0.114, 0.404, 0.178],
        ["Need for digitalis", "Logistic Regression", 0.765, 0.842, 0.634, 0.723],
        ["HFrEF before echo", "Logistic Regression", 0.759, 0.480, 0.462, 0.471],
        ["Need for IV inotropes", "Random Forest", 0.754, 0.638, 0.530, 0.579],
        ["6-month death: severity only", "Logistic Regression", 0.745, 0.095, 0.333, 0.147],
        ["6-month readmission (best)", "Random Forest", 0.666, None, None, None],
        ["28-day readmission", "Random Forest", 0.664, None, None, None],
    ], columns=["Question", "Model", "ROC-AUC", "Precision", "Recall", "F1"])
    c1, c2 = st.columns([3, 2])
    with c1:
        lbs = lb.sort_values('ROC-AUC')
        colors = ['#bdbbb4' if v < 0.7 else (RED if 'death' in q.lower() else BLUE)
                  for q, v in zip(lbs['Question'], lbs['ROC-AUC'])]
        fig = go.Figure(go.Bar(x=lbs['ROC-AUC'], y=lbs['Question'], orientation='h', marker_color=colors,
                               text=[f"{v:.3f}" for v in lbs['ROC-AUC']], textposition='outside',
                               customdata=lbs['Model'], hovertemplate="%{y}<br>%{customdata}: AUC %{x:.3f}<extra></extra>"))
        for x, lab in [(0.7, 'acceptable'), (0.8, 'good'), (0.9, 'excellent')]:
            fig.add_vline(x=x, line_dash='dot', line_color=INK2, annotation_text=lab, annotation_position='top')
        fig.update_layout(title="Model leaderboard (ROC-AUC)",
                          xaxis_range=[0.5, 1.0], xaxis_title="Cross-validated ROC-AUC", yaxis_title="")
        show(style(fig, 470, legend=False), "Cross-validated ROC-AUC of every model (0.5 = chance, 1 = perfect). Red = mortality "
                                            "models, blue = phenotype/treatment models, grey = readmission models, which "
                                            "stay below the 0.7 'acceptable' line.")
    with c2:
        table(lb.style.format({"ROC-AUC": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}, na_rep="–"),
                  hide_index=True, height=470)
    insight("<b>Mortality is predictable from admission data</b> (AUC 0.90 at 28 days, fading to 0.78 by 6 months). "
            "<b>Readmission is not</b> (AUC ≤ 0.67) — it depends on things this dataset does not record.")

    st.markdown("---")
    st.markdown("### Interactive model explorer")
    st.caption("Trains the selected model on all patients with 5-fold cross-validation (first run ~10–40 s, then cached). "
               "Move the slider to change how many patients are flagged and watch the confusion matrix and metrics update. "
               "(One 5-fold run, so ROC-AUC can differ from the leaderboard by about ±0.01.)")
    c1, c2 = st.columns([2, 3])
    with c1:
        mkey = st.selectbox("Model", ["28-day death (full admission data)", "3-month death (full admission data)",
                                      "6-month death (full admission data)", "28-day death (severity + GCS, simple)",
                                      "HFrEF before echo (10 bedside features)", "Need for IV inotropes",
                                      "Need for digitalis (digoxin/deslanoside)"])
    with st.spinner("Training and cross-validating the model…"):
        y, p, imp, kind, top_default = run_model(mkey, df_all, len(df_all))
    with c2:
        top = st.slider("Flag the top X% highest predicted risk", 1, 60, int(top_default * 100), step=1) / 100

    is_death = 'death' in mkey.lower()
    tcol = DEATH_C if is_death else NEUTRAL
    tramp = DEATH_RAMP if is_death else ["#e3e7ec", "#c2cad4", "#9aa6b5", "#6f7f93", "#4b5d73", "#2c3a4b"]
    thr = np.quantile(p, 1 - top)
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    m = st.columns(6)
    m[0].metric("ROC-AUC", f"{roc_auc_score(y, p):.3f}")
    m[1].metric("PR-AUC", f"{average_precision_score(y, p):.3f}", f"chance {y.mean():.3f}", delta_color="off")
    m[2].metric("Accuracy", f"{accuracy_score(y, pred):.3f}")
    m[3].metric("Precision", f"{precision_score(y, pred, zero_division=0):.3f}")
    m[4].metric("Recall", f"{recall_score(y, pred):.3f}")
    m[5].metric("F1-score", f"{f1_score(y, pred, zero_division=0):.3f}")

    c1, c2, c3 = st.columns([1.2, 1, 1.2])
    with c1:
        fpr, tpr, thr_all = roc_curve(y, p)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', line=dict(color=BLUE, width=2.5), name='Model',
                                 hovertemplate="False-positive rate %{x:.2f}<br>Recall %{y:.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color='#bdbbb4', dash='dash'), name='Chance'))
        cur_fpr, cur_tpr = fp / (fp + tn) if (fp + tn) else 0, tp / (tp + fn) if (tp + fn) else 0
        fig.add_trace(go.Scatter(x=[cur_fpr], y=[cur_tpr], mode='markers', marker=dict(size=13, color=RED, line=dict(color='white', width=2)),
                                 name='Current cut-off'))
        fig.update_layout(title="ROC curve", xaxis_title="False-positive rate", yaxis_title="Recall")
        fig = style(fig, 380); fig.update_layout( legend=dict(orientation='v', yanchor='bottom', y=0.02, xanchor='right', x=0.98, bgcolor='rgba(255,255,255,0.8)'))
        show(fig, f"Share of events caught (recall) against share of non-events wrongly flagged, for every cut-off "
                  f"(AUC {roc_auc_score(y, p):.3f}). The red dot is the current slider setting; the dashed line is chance.")
    with c2:
        fig, ax = plt.subplots(figsize=(3.3, 3.4))
        cm = np.array([[tn, fp], [fn, tp]])
        from matplotlib.colors import LinearSegmentedColormap
        ax.imshow(cm, cmap=SOFT_BLUES)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{['TN', 'FP', 'FN', 'TP'][2*i+j]}\n{cm[i, j]}", ha='center', va='center', fontsize=12,
                        fontweight='bold', color='white' if cm[i, j] > cm.max() / 2 else INK)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['Not flagged', 'Flagged'])
        ax.set_yticks([0, 1]); ax.set_yticklabels(['No event', 'Event'])
        ax.set_xlabel("Model decision", fontsize=11, color='#000000')
        ax.set_ylabel("Actual outcome", fontsize=11, color='#000000')
        mpl_style(ax); ax.tick_params(length=0); plt.tight_layout()
        show_mpl(fig, title="Confusion matrix", caption=f"Counts when the top {top:.0%} highest-risk patients are flagged: TN = correctly not flagged, "
                      f"FP = false alarms, FN = missed events, TP = events caught.")
    with c3:
        top_imp = imp.reindex(imp.abs().sort_values(ascending=False).index).head(12)[::-1]
        lab = "Importance" if kind == "rf" else "Coefficient (+ raises risk)"
        colr = [BLUE if kind == 'rf' else (RED if v > 0 else AQUA) for v in top_imp.values]
        fig = go.Figure(go.Bar(x=top_imp.values, y=top_imp.index, orientation='h', marker_color=colr,
                               hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        fig.update_layout(title="Top predictors", xaxis_title=lab, yaxis_title="")
        show(style(fig, 380, legend=False), "The 12 features the model relies on most. For Random Forest, a longer bar = more "
                                            "useful; for Logistic Regression, red bars raise risk and aqua bars lower it.")

    c1, c2 = st.columns([1, 1])
    with c1:
        dec = pd.qcut(pd.Series(p).rank(method='first'), 10, labels=range(1, 11))
        dr = (pd.Series(y).groupby(dec.values).mean() * 100).reset_index()
        dr.columns = ['Risk decile', 'Observed rate %']
        fig = px.bar(dr, x='Risk decile', y='Observed rate %', text=dr['Observed rate %'].round(1).astype(str) + '%',
                     title="Observed rate by risk decile")
        fig.update_traces(marker_color=[READM_RAMP[min(5, i // 2)] for i in range(10)], textposition='outside')
        fig.update_layout(xaxis=dict(dtick=1))
        show(style(fig, 340, legend=False), "Patients sorted into 10 equal groups by predicted risk (1 = lowest, 10 = highest), "
                                            "with the observed event rate in each. A steep rise at the right means the model "
                                            "ranks patients well.")
    with c2:
        st.markdown("**Classification report**")
        rep = pd.DataFrame(classification_report(y, pred, labels=[0, 1], target_names=['No event (0)', 'Event (1)'],
                                                 output_dict=True, zero_division=0)).T
        rep.loc['accuracy', 'support'] = len(y)
        rep['support'] = rep['support'].round(0).astype(int)
        table(rep.style.format({'precision': '{:.3f}', 'recall': '{:.3f}', 'f1-score': '{:.3f}', 'support': '{:,}'}))
        st.caption(f"{int(pred.sum())} of {len(y)} patients flagged · {int(tp)} of {int(y.sum())} events caught · "
                   f"{tp/pred.sum()*100 if pred.sum() else 0:.1f}% of flagged patients had the event "
                   f"(base rate {y.mean()*100:.1f}%)")
    if y.mean() < 0.1:
        caution("Events are rare, so accuracy is high even for weak models. Judge the model by ROC-AUC, recall and how many "
                "times precision exceeds the base rate.")

    st.markdown("---")
    st.markdown("### Bedside risk calculator — 28-day death")
    st.caption("Logistic regression on cardiac severity + consciousness (Question 3 model, AUC 0.84), fitted on all patients. "
               "For education only — not for clinical decisions.")

    @st.cache_resource(show_spinner=False)
    def calc_model(n):
        mdl = logreg().fit(df_all[SEVERITY + GCS], df_all['death_28d'])
        base = logreg()
        raw = cross_val_predict(base, df_all[SEVERITY + GCS], df_all['death_28d'],
                                cv=StratifiedKFold(5, shuffle=True, random_state=0), method='predict_proba')[:, 1]
        return mdl, raw

    cm_model, cv_scores = calc_model(len(df_all))
    c1, c2, c3 = st.columns(3)
    with c1:
        nyha = st.select_slider("NYHA class", [2, 3, 4], value=3)
        killip = st.select_slider("Killip class", [1, 2, 3, 4], value=2)
        severe = st.toggle("Severe HF flag", value=False, help="Severe breathlessness and severe congestion/shock at admission")
    with c2:
        eye = st.slider("Eye opening (1–4)", 1, 4, 4)
        verbal = st.slider("Verbal response (1–5)", 1, 5, 5)
        move = st.slider("Movement (1–6)", 1, 6, 6)
    g = eye + verbal + move
    row = pd.DataFrame([[nyha, killip, int(severe), g, eye, verbal, move, int(g == 15)]], columns=SEVERITY + GCS)
    score = cm_model.predict_proba(row)[0, 1]
    pctile = (cv_scores < score).mean() * 100
    group = cv_scores >= np.quantile(cv_scores, 0.9)
    with c3:
        band = "High" if pctile >= 90 else "Moderate" if pctile >= 60 else "Low"
        colr = RED if band == "High" else ORANGE if band == "Moderate" else AQUA
        st.markdown(f'<div class="kpi" style="border-top:4px solid {colr}"><div class="lbl">Risk position (GCS {g})</div>'
                    f'<div class="val">{band}</div><div class="sub">Higher than {pctile:.0f}% of patients in the cohort<br>'
                    f'Patients in the top 10% had 28-day death of {df_all["death_28d"][group].mean()*100:.1f}% '
                    f'vs {df_all["death_28d"][~group].mean()*100:.1f}% for the rest</div></div>', unsafe_allow_html=True)
        st.caption("The model uses class-balanced weights, so its raw score is a ranking, not a calibrated probability — "
                   "the cohort percentile is the meaningful number.")
