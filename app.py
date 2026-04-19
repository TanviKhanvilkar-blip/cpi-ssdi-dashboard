import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, t
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
import sklearn.linear_model as lm
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="India CPI — SSDI Project",
    page_icon="📊",
    layout="wide"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem; font-weight: 700; color: #1a3a5c;
        text-align: center; padding: 0.8rem 0 0.2rem 0;
    }
    .sub-header {
        font-size: 0.95rem; color: #666;
        text-align: center; margin-bottom: 1rem;
    }
    .section-title {
        font-size: 1.35rem; font-weight: 600; color: #1a3a5c;
        border-left: 5px solid #e74c3c; padding-left: 10px; margin: 0.8rem 0;
    }
    .reject-box {
        background: #fdecea; border-left: 4px solid #e74c3c;
        padding: 0.7rem 1rem; border-radius: 5px; margin: 0.5rem 0;
    }
    .accept-box {
        background: #eaf4ea; border-left: 4px solid #27ae60;
        padding: 0.7rem 1rem; border-radius: 5px; margin: 0.5rem 0;
    }
    .info-box {
        background: #eaf0fb; border-left: 4px solid #2980b9;
        padding: 0.7rem 1rem; border-radius: 5px; margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
X_COLS   = ['Food_and_Beverages', 'Fuel_and_Light', 'Housing',
            'Clothing_and_Footwear', 'Miscellaneous']
X_LABELS = ['Food & Beverages', 'Fuel & Light', 'Housing',
            'Clothing & Footwear', 'Miscellaneous']
COLORS   = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

# ── Data loading — already clean, just read and fix dtypes ───────────────────
@st.cache_data
def load_data():
    df = pd.read_excel("cpi_clean.xlsx")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['year', 'month_code']).reset_index(drop=True)
    return df

try:
    df = load_data()
    DATA_OK = True
except Exception as e:
    DATA_OK = False
    ERR = str(e)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">📊 What Drives Consumer Price Inflation in India?</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Statistical Analysis of CPI (2015–2025) &nbsp;|&nbsp; '
    'MBATech Data Science SEM IV &nbsp;|&nbsp; SSDI Project &nbsp;|&nbsp; MPSTME NMIMS'
    '</div>', unsafe_allow_html=True)
st.markdown("---")

if not DATA_OK:
    st.error(f"Cannot load `cpi_clean.xlsx`. Make sure it is in the same folder as `app.py`.\n\nError: {ERR}")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 📌 Navigation")
section = st.sidebar.radio("", [
    "🏠 Overview & EDA",
    "🧪 Hypothesis Testing",
    "📊 ANOVA",
    "📈 OLS Linear Regression",
    "🔵 Ridge & Lasso Regression",
])
st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** MoSPI eSankhyiki")
st.sidebar.markdown("**Series:** CPI Base 2012 = 100")
st.sidebar.markdown("**Period:** Jan 2015 – Dec 2025")
st.sidebar.markdown(f"**Records:** {len(df)} monthly observations")
st.sidebar.markdown("**Sector:** All India, Combined")
st.sidebar.markdown("[🔗 Data Source](https://esankhyiki.mospi.gov.in)")

# ═══════════════════════════════════════════════════════════════════════════════
# 1 — OVERVIEW & EDA
# ═══════════════════════════════════════════════════════════════════════════════
if section == "🏠 Overview & EDA":
    st.markdown('<div class="section-title">Overview & Exploratory Data Analysis</div>',
                unsafe_allow_html=True)

    yr_min, yr_max = int(df['year'].min()), int(df['year'].max())
    y1, y2 = st.slider("Filter Year Range", yr_min, yr_max, (yr_min, yr_max))
    dff = df[(df['year'] >= y1) & (df['year'] <= y2)].copy()

    pre  = dff[dff['covid_period'] == 'Pre-COVID']['General']
    post = dff[dff['covid_period'] == 'Post-COVID']['General']

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Overall Mean CPI", f"{dff['General'].mean():.1f}")
    c2.metric("Pre-COVID Mean",   f"{pre.mean():.1f}"  if len(pre)  else "N/A")
    c3.metric("Post-COVID Mean",  f"{post.mean():.1f}" if len(post) else "N/A",
              delta=f"+{post.mean()-pre.mean():.1f}" if len(pre) and len(post) else None)
    c4.metric("Max CPI",          f"{dff['General'].max():.1f}")
    c5.metric("Max YoY Inflation",
              f"{dff['yoy_inflation'].max():.1f}%" if dff['yoy_inflation'].notna().any() else "N/A")

    st.markdown("---")

    # General CPI Trend
    st.markdown("#### 📉 General CPI Trend")
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(dff['date'], dff['General'], color='#1a3a5c', linewidth=2, label='General CPI')
    if dff['date'].min() < pd.Timestamp('2020-01-01') < dff['date'].max():
        ax.axvline(pd.Timestamp('2020-01-01'), color='red', linestyle='--',
                   linewidth=1.5, label='COVID Start (Jan 2020)')
        ax.fill_between(dff['date'], dff['General'].min(), dff['General'],
                        where=dff['date'] >= pd.Timestamp('2020-01-01'),
                        alpha=0.08, color='red')
    ax.set_ylabel('CPI Index (Base 2012=100)')
    ax.set_xlabel('Date')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title('All India General CPI — Monthly', fontsize=12)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Sub-group trends
    st.markdown("#### 📈 CPI Sub-Group Trends")
    selected_groups = st.multiselect(
        "Select sub-groups to display", options=X_COLS, default=X_COLS,
        format_func=lambda x: x.replace('_', ' ')
    )
    if selected_groups:
        fig, ax = plt.subplots(figsize=(12, 3.5))
        for col, color in zip(X_COLS, COLORS):
            if col in selected_groups:
                ax.plot(dff['date'], dff[col],
                        label=col.replace('_', ' '), color=color, linewidth=1.6)
        if dff['date'].min() < pd.Timestamp('2020-01-01') < dff['date'].max():
            ax.axvline(pd.Timestamp('2020-01-01'), color='black', linestyle='--',
                       linewidth=1.2, label='COVID Start')
        ax.set_ylabel('CPI Index')
        ax.set_xlabel('Date')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_title('CPI by Sub-Group — Monthly Trends', fontsize=12)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # Correlation + Descriptive stats
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🔗 Correlation Matrix")
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        sns.heatmap(dff[X_COLS + ['General']].corr(), annot=True, cmap='coolwarm',
                    fmt='.2f', ax=ax, linewidths=0.5, square=True, vmin=-1, vmax=1)
        ax.set_title('Correlation Matrix', fontsize=11)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        st.markdown("#### 📋 Descriptive Statistics")
        desc = dff[['General'] + X_COLS].describe().round(2)
        desc.index = ['Count', 'Mean', 'Std', 'Min', '25%', 'Median', '75%', 'Max']
        st.dataframe(desc, use_container_width=True)

    st.markdown("#### 🗂 Data Preview")
    st.dataframe(
        dff[['date', 'year', 'month', 'General'] + X_COLS + ['covid_period', 'yoy_inflation']],
        use_container_width=True, height=250
    )

# ═══════════════════════════════════════════════════════════════════════════════
# 2 — HYPOTHESIS TESTING
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "🧪 Hypothesis Testing":
    st.markdown('<div class="section-title">Hypothesis Testing</div>', unsafe_allow_html=True)

    test_choice = st.selectbox("Select Test", [
        "Two-Sample t-Test: Pre-COVID vs Post-COVID CPI",
        "One-Sample t-Test: Test a hypothesised mean CPI",
        "Z-Test: Large sample — Pre vs Post COVID"
    ])
    alpha = st.selectbox("Significance Level (α)", [0.01, 0.05, 0.10], index=1)
    tail  = st.radio("Tail", ["Two-tailed", "One-tailed (right)", "One-tailed (left)"],
                     horizontal=True)

    pre_data  = df[df['covid_period'] == 'Pre-COVID']['General'].values
    post_data = df[df['covid_period'] == 'Post-COVID']['General'].values
    st.markdown("---")

    # ── Two-sample t-test ──────────────────────────────────────────────────────
    if test_choice == "Two-Sample t-Test: Pre-COVID vs Post-COVID CPI":
        st.markdown("""
        **Hypotheses:**
        - **H₀ :** μ(Post-COVID) = μ(Pre-COVID)
        - **H₁ :** μ(Post-COVID) > μ(Pre-COVID)
        """)

        n1, n2   = len(post_data), len(pre_data)
        xb1, xb2 = np.mean(post_data), np.mean(pre_data)
        s1,  s2  = np.std(post_data, ddof=1), np.std(pre_data, ddof=1)
        sp   = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
        se   = sp * np.sqrt(1/n1 + 1/n2)
        t_cal = (xb1 - xb2) / se
        df_t  = n1 + n2 - 2

        if tail == "Two-tailed":
            t_crit = t.ppf(1 - alpha/2, df_t)
            p_val  = 2 * (1 - t.cdf(abs(t_cal), df_t))
        elif tail == "One-tailed (right)":
            t_crit = t.ppf(1 - alpha, df_t)
            p_val  = 1 - t.cdf(t_cal, df_t)
        else:
            t_crit = t.ppf(alpha, df_t)
            p_val  = t.cdf(t_cal, df_t)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pre-COVID Mean",  f"{xb2:.2f}")
        c2.metric("Post-COVID Mean", f"{xb1:.2f}")
        c3.metric("t-calculated",    f"{t_cal:.4f}")
        c4.metric("p-value",         f"{p_val:.6f}")
        st.metric("t-critical",
                  f"±{t_crit:.4f}" if tail == "Two-tailed" else f"{t_crit:.4f}")

        if p_val < alpha:
            st.markdown(
                f'<div class="reject-box">🔴 <b>Reject H₀</b> — Post-COVID CPI is significantly '
                f'higher (p = {p_val:.4f} &lt; α = {alpha})</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="accept-box">🟢 <b>Fail to Reject H₀</b> — No significant difference '
                f'(p = {p_val:.4f} ≥ α = {alpha})</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            x_r = np.linspace(-5, 8, 400)
            ax.plot(x_r, t.pdf(x_r, df_t), 'steelblue', linewidth=2, label='t-distribution')
            ax.axvline(t_cal, color='red', linestyle='--', linewidth=2,
                       label=f't-cal = {t_cal:.3f}')
            ax.axvline(abs(t_crit), color='green', linestyle='--', linewidth=2,
                       label=f't-critical = {abs(t_crit):.3f}')
            if tail == "One-tailed (right)":
                xf = np.linspace(t_crit, 8, 100)
                ax.fill_between(xf, t.pdf(xf, df_t), alpha=0.25, color='red',
                                label='Rejection Region')
            elif tail == "One-tailed (left)":
                xf = np.linspace(-5, t_crit, 100)
                ax.fill_between(xf, t.pdf(xf, df_t), alpha=0.25, color='red',
                                label='Rejection Region')
            else:
                ax.fill_between(np.linspace(-5, -abs(t_crit), 100),
                                t.pdf(np.linspace(-5, -abs(t_crit), 100), df_t),
                                alpha=0.25, color='red')
                xf = np.linspace(abs(t_crit), 8, 100)
                ax.fill_between(xf, t.pdf(xf, df_t), alpha=0.25, color='red',
                                label='Rejection Region')
            ax.set_title(f't-Distribution | df={df_t}', fontsize=11)
            ax.set_xlabel('t value'); ax.set_ylabel('Density')
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.boxplot([pre_data, post_data], labels=['Pre-COVID', 'Post-COVID'],
                       patch_artist=True,
                       boxprops=dict(facecolor='steelblue', alpha=0.5),
                       medianprops=dict(color='red', linewidth=2))
            ax.set_ylabel('General CPI Index')
            ax.set_title('CPI Distribution: Pre vs Post COVID')
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── One-sample t-test ──────────────────────────────────────────────────────
    elif test_choice == "One-Sample t-Test: Test a hypothesised mean CPI":
        mu0 = st.number_input("Hypothesised population mean (μ₀)", value=150.0, step=1.0)
        st.markdown(f"""
        **Hypotheses:**
        - **H₀ :** μ = {mu0}
        - **H₁ :** μ ≠ {mu0}
        """)
        X_arr = df['General'].values
        n     = len(X_arr)
        xbar  = np.mean(X_arr)
        SD    = np.std(X_arr, ddof=1)
        se    = SD / np.sqrt(n)
        t_cal = (xbar - mu0) / se
        df_t  = n - 1
        t_crit = t.ppf(1 - alpha/2, df_t) if tail == "Two-tailed" \
            else t.ppf(1 - alpha, df_t)

        alt_map = {"Two-tailed": "two-sided",
                   "One-tailed (right)": "greater",
                   "One-tailed (left)": "less"}
        lib_res = stats.ttest_1samp(X_arr, popmean=mu0, alternative=alt_map[tail])

        c1, c2, c3 = st.columns(3)
        c1.metric("Sample Mean",  f"{xbar:.2f}")
        c2.metric("t-calculated", f"{t_cal:.4f}")
        c3.metric("p-value",      f"{lib_res.pvalue:.6f}")
        st.metric("t-critical",
                  f"±{t_crit:.4f}" if tail == "Two-tailed" else f"{t_crit:.4f}")

        if lib_res.pvalue < alpha:
            st.markdown(
                f'<div class="reject-box">🔴 <b>Reject H₀</b> — Mean CPI is significantly '
                f'different from {mu0}</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="accept-box">🟢 <b>Fail to Reject H₀</b> — Mean CPI not '
                f'significantly different from {mu0}</div>', unsafe_allow_html=True)

    # ── Z-test ─────────────────────────────────────────────────────────────────
    else:
        st.markdown("""
        **Hypotheses (Large Sample Z-Test):**
        - **H₀ :** μ(Post-COVID) = μ(Pre-COVID)
        - **H₁ :** μ(Post-COVID) > μ(Pre-COVID)
        """)
        n1, n2   = len(post_data), len(pre_data)
        xb1, xb2 = np.mean(post_data), np.mean(pre_data)
        s1,  s2  = np.std(post_data, ddof=1), np.std(pre_data, ddof=1)
        se    = np.sqrt(s1**2/n1 + s2**2/n2)
        z_cal = (xb1 - xb2) / se

        if tail == "Two-tailed":
            z_crit = norm.ppf(1 - alpha/2)
            p_val  = 2 * (1 - norm.cdf(abs(z_cal)))
        elif tail == "One-tailed (right)":
            z_crit = norm.ppf(1 - alpha)
            p_val  = 1 - norm.cdf(z_cal)
        else:
            z_crit = norm.ppf(alpha)
            p_val  = norm.cdf(z_cal)

        c1, c2, c3 = st.columns(3)
        c1.metric("z-calculated", f"{z_cal:.4f}")
        c2.metric("z-critical",   f"{z_crit:.4f}")
        c3.metric("p-value",      f"{p_val:.6f}")

        if p_val < alpha:
            st.markdown(
                f'<div class="reject-box">🔴 <b>Reject H₀</b> — Significant difference '
                f'(p = {p_val:.4f} &lt; α = {alpha})</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="accept-box">🟢 <b>Fail to Reject H₀</b></div>',
                unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(10, 4))
        x_r = np.linspace(-4, 7, 400)
        ax.plot(x_r, norm.pdf(x_r), 'steelblue', linewidth=2, label='Standard Normal')
        ax.axvline(z_cal,  color='red',   linestyle='--', linewidth=2,
                   label=f'z-cal = {z_cal:.3f}')
        ax.axvline(z_crit, color='green', linestyle='--', linewidth=2,
                   label=f'z-critical = {z_crit:.3f}')
        if tail == "One-tailed (right)":
            xf = np.linspace(z_crit, 7, 100)
            ax.fill_between(xf, norm.pdf(xf), alpha=0.25, color='red',
                            label='Rejection Region')
        ax.set_title(f'Z-Test Distribution | α={alpha}', fontsize=12)
        ax.set_xlabel('z value'); ax.set_ylabel('Density')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 3 — ANOVA
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "📊 ANOVA":
    st.markdown('<div class="section-title">ANOVA — Analysis of Variance</div>',
                unsafe_allow_html=True)

    anova_type = st.radio(
        "Select ANOVA Type",
        ["One-Way ANOVA", "Two-Way ANOVA (with Interaction)", "Tukey HSD Post-Hoc Test"],
        horizontal=True
    )
    alpha_a = st.selectbox("α level", [0.01, 0.05, 0.10], index=1)

    df_melt = df[X_COLS + ['covid_period', 'year']].melt(
        id_vars=['covid_period', 'year'], var_name='Group', value_name='CPI_Index'
    )
    df_melt['Group'] = df_melt['Group'].str.replace('_', ' ')
    df_melt = df_melt.dropna(subset=['CPI_Index'])
    st.markdown("---")

    if anova_type == "One-Way ANOVA":
        st.markdown("""
        **Hypotheses:**
        - **H₀ :** Mean CPI is equal across all sub-groups
        - **H₁ :** At least one group mean is significantly different
        """)
        f_stat, p_val = stats.f_oneway(*[df[c].dropna().values for c in X_COLS])
        fit1      = ols('CPI_Index ~ C(Group)', data=df_melt).fit()
        anova_tbl = sm.stats.anova_lm(fit1, typ=1)

        c1, c2 = st.columns(2)
        c1.metric("F-statistic", f"{f_stat:.4f}")
        c2.metric("p-value",     f"{p_val:.6f}")

        if p_val < alpha_a:
            st.markdown(
                f'<div class="reject-box">🔴 <b>Reject H₀</b> — Significant differences '
                f'between group means (p &lt; {alpha_a})</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="accept-box">🟢 <b>Fail to Reject H₀</b></div>',
                unsafe_allow_html=True)

        st.markdown("**ANOVA Table (statsmodels):**")
        st.dataframe(anova_tbl.round(4), use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 4))
        sns.boxplot(data=df_melt, x='Group', y='CPI_Index', palette='Set2', ax=ax)
        ax.set_title(f'CPI by Sub-Group  |  F = {f_stat:.2f},  p = {p_val:.4f}', fontsize=12)
        ax.set_xlabel(''); ax.set_ylabel('CPI Index')
        plt.xticks(rotation=12); ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    elif anova_type == "Two-Way ANOVA (with Interaction)":
        st.markdown("""
        **Testing Group × COVID Period interaction on CPI Index**
        - **H₀ :** No significant interaction between Group and COVID period
        - **H₁ :** Interaction is significant
        """)
        fit2   = ols('CPI_Index ~ C(Group) * C(covid_period)', data=df_melt).fit()
        anova2 = sm.stats.anova_lm(fit2, typ=2)

        st.markdown("**Two-Way ANOVA Table:**")
        st.dataframe(anova2.round(4), use_container_width=True)

        sig = anova2[anova2['PR(>F)'] < alpha_a].index.tolist()
        if sig:
            st.markdown(
                f'<div class="reject-box">🔴 Significant effects found for: '
                f'<b>{", ".join(sig)}</b></div>', unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(11, 4))
        sns.boxplot(data=df_melt, x='Group', y='CPI_Index', hue='covid_period',
                    palette={'Pre-COVID': 'steelblue', 'Post-COVID': 'tomato'}, ax=ax)
        ax.set_title('CPI by Sub-Group × COVID Period', fontsize=12)
        ax.set_xlabel(''); ax.set_ylabel('CPI Index')
        plt.xticks(rotation=12)
        ax.legend(title='Period', fontsize=9); ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    else:  # Tukey
        st.markdown("**Tukey's HSD** — identifies which specific sub-group pairs differ significantly")
        tukey    = pairwise_tukeyhsd(df_melt['CPI_Index'], groups=df_melt['Group'])
        tukey_df = pd.DataFrame(
            data=tukey._results_table.data[1:],
            columns=tukey._results_table.data[0]
        ).sort_values('meandiff', ascending=False)
        rejected = tukey_df[tukey_df['reject'] == True]
        st.markdown(f"**{len(rejected)} of {len(tukey_df)} pairs** show significant differences:")
        st.dataframe(
            tukey_df.style.apply(
                lambda row: ['background-color:#fdecea' if row['reject'] else '' for _ in row],
                axis=1),
            use_container_width=True
        )

        group_means = df_melt.groupby('Group')['CPI_Index'].mean().sort_values()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(group_means.index, group_means.values, color='steelblue', alpha=0.8)
        ax.set_xlabel('Mean CPI Index')
        ax.set_title('Mean CPI by Sub-Group', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 4 — OLS LINEAR REGRESSION
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "📈 OLS Linear Regression":
    st.markdown('<div class="section-title">Multiple Linear Regression (OLS)</div>',
                unsafe_allow_html=True)
    st.markdown("Predicting **General CPI** from sub-group indices using `smf.ols()`")

    selected_x = st.multiselect(
        "Select predictor variables (X)", options=X_COLS, default=X_COLS,
        format_func=lambda x: x.replace('_', ' ')
    )
    if not selected_x:
        st.warning("Please select at least one predictor."); st.stop()

    df_reg  = df[selected_x + ['General']].dropna()
    formula = 'General ~ ' + ' + '.join(selected_x)
    fit     = smf.ols(formula, data=df_reg).fit()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R²",          f"{fit.rsquared:.4f}")
    c2.metric("Adj. R²",     f"{fit.rsquared_adj:.4f}")
    c3.metric("F-statistic", f"{fit.fvalue:.2f}")
    c4.metric("F p-value",   f"{fit.f_pvalue:.6f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Regression Coefficients:**")
        coef_df = pd.DataFrame({
            'Coefficient': fit.params,
            'Std Error':   fit.bse,
            't-value':     fit.tvalues,
            'p-value':     fit.pvalues,
            'CI Lower':    fit.conf_int()[0],
            'CI Upper':    fit.conf_int()[1]
        }).round(4)
        st.dataframe(coef_df, use_container_width=True)

    with col2:
        st.markdown("**Variance Inflation Factors (VIF):**")
        X_vif  = df_reg[selected_x]
        vif_df = pd.DataFrame({
            'Feature': selected_x,
            'VIF':     [variance_inflation_factor(X_vif.values, i)
                        for i in range(len(selected_x))]
        }).round(2)
        st.dataframe(vif_df, use_container_width=True)
        if vif_df['VIF'].max() > 10:
            st.markdown(
                '<div class="reject-box">⚠️ High VIF (&gt;10) detected — multicollinearity '
                'present. This motivates Ridge &amp; Lasso.</div>', unsafe_allow_html=True)

    # Diagnostic plots
    fitted    = fit.fittedvalues
    residuals = fit.resid
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(fitted, residuals, alpha=0.5, color='steelblue', s=25)
    axes[0].axhline(0, color='red', linestyle='--', linewidth=1.5)
    axes[0].set_title('Residuals vs Fitted Values')
    axes[0].set_xlabel('Fitted Values'); axes[0].set_ylabel('Residuals')
    axes[0].grid(True, alpha=0.3)
    sm.qqplot(residuals, line='s', ax=axes[1], alpha=0.5)
    axes[1].set_title('Q-Q Plot of Residuals')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Actual vs Predicted
    st.markdown("#### Actual vs Predicted General CPI")
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(df_reg.index, df_reg['General'],
            label='Actual',    color='steelblue', linewidth=1.8)
    ax.plot(df_reg.index, fitted,
            label='Predicted', color='red', linestyle='--', linewidth=1.5, alpha=0.85)
    ax.set_ylabel('General CPI Index'); ax.set_xlabel('Observation Index')
    ax.set_title('Actual vs Predicted General CPI (OLS)', fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 5 — RIDGE & LASSO
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "🔵 Ridge & Lasso Regression":
    st.markdown('<div class="section-title">Ridge & Lasso Regression</div>',
                unsafe_allow_html=True)
    st.markdown("Regularization to handle multicollinearity — coefficient paths + KFold CV")

    k_folds = st.slider("KFold splits (K)", 3, 15, 10)

    df_reg = df[X_COLS + ['General']].dropna()
    X      = df_reg[X_COLS].values
    Y      = df_reg['General'].values
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    alphas = 10**np.linspace(10, -2, 100) * 0.5

    # Coefficient paths
    ridge_coefs, lasso_coefs = [], []
    for a in alphas:
        r = Ridge(alpha=a); r.fit(X_sc, Y); ridge_coefs.append(r.coef_)
        try:
            l = Lasso(alpha=a, max_iter=10000); l.fit(X_sc, Y); lasso_coefs.append(l.coef_)
        except Exception:
            lasso_coefs.append([0] * len(X_COLS))

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, (label, color) in enumerate(zip(X_LABELS, COLORS)):
            ax.plot(alphas, [c[i] for c in ridge_coefs],
                    color=color, label=label, linewidth=1.6)
        ax.set_xscale('log'); ax.set_xlabel('Alpha (λ)'); ax.set_ylabel('Coefficient')
        ax.set_title('Ridge — Coefficient Path')
        ax.legend(fontsize=7, loc='upper right'); ax.grid(True, alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, (label, color) in enumerate(zip(X_LABELS, COLORS)):
            ax.plot(alphas, [c[i] for c in lasso_coefs],
                    color=color, label=label, linewidth=1.6)
        ax.set_xscale('log'); ax.set_xlabel('Alpha (λ)'); ax.set_ylabel('Coefficient')
        ax.set_title('Lasso — Coefficient Path (Features Zeroed Out)')
        ax.legend(fontsize=7, loc='upper right'); ax.grid(True, alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown(f"---\n#### Cross-Validation (KFold = {k_folds}) — Best Alpha Selection")
    kfold   = KFold(k_folds, random_state=0, shuffle=True)
    ridgeCV = RidgeCV(alphas=alphas, cv=kfold); ridgeCV.fit(X_sc, Y)
    lassoCV = LassoCV(alphas=alphas, cv=kfold, max_iter=10000); lassoCV.fit(X_sc, Y)

    lin_model = lm.LinearRegression()
    lin_mse   = -cross_val_score(lin_model, X_sc, Y, cv=kfold,
                                  scoring='neg_mean_squared_error').mean()
    lin_model.fit(X_sc, Y)
    ridge_mse = np.mean((Y - ridgeCV.predict(X_sc))**2)
    lasso_mse = np.mean((Y - lassoCV.predict(X_sc))**2)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**📌 OLS Baseline**")
        st.metric("CV MSE", f"{lin_mse:.4f}")
    with c2:
        st.markdown("**📌 Ridge**")
        st.metric("Best Alpha", f"{ridgeCV.alpha_:.4f}")
        st.metric("R²",         f"{ridgeCV.score(X_sc, Y):.4f}")
        st.metric("MSE",        f"{ridge_mse:.4f}")
    with c3:
        st.markdown("**📌 Lasso**")
        st.metric("Best Alpha", f"{lassoCV.alpha_:.4f}")
        st.metric("R²",         f"{lassoCV.score(X_sc, Y):.4f}")
        st.metric("MSE",        f"{lasso_mse:.4f}")

    st.markdown("#### Coefficient Comparison: OLS vs Ridge vs Lasso")
    coef_df = pd.DataFrame({
        'Feature': X_LABELS,
        'OLS':     lin_model.coef_,
        'Ridge':   ridgeCV.coef_,
        'Lasso':   lassoCV.coef_
    }).round(4)
    st.dataframe(coef_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    x_pos = np.arange(len(X_LABELS))
    ax.bar(x_pos - 0.25, lin_model.coef_, 0.25, label='OLS',   color='#95a5a6')
    ax.bar(x_pos,        ridgeCV.coef_,   0.25, label='Ridge', color='steelblue')
    ax.bar(x_pos + 0.25, lassoCV.coef_,   0.25, label='Lasso', color='tomato')
    ax.set_xticks(x_pos); ax.set_xticklabels(X_LABELS, rotation=12)
    ax.set_ylabel('Coefficient (Standardized)')
    ax.set_title('OLS vs Ridge vs Lasso — Coefficient Comparison')
    ax.legend(); ax.axhline(0, color='black', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    zeroed = [X_LABELS[i] for i, c in enumerate(lassoCV.coef_) if abs(c) < 0.001]
    if zeroed:
        st.markdown(
            f'<div class="accept-box">✅ <b>Lasso zeroed out:</b> {", ".join(zeroed)} '
            f'— weak predictors of General CPI</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="info-box">ℹ️ Lasso retained all features at the selected alpha. '
            'Try increasing K.</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#999;font-size:0.82rem;'>"
    "SSDI Project | MBATech Data Science SEM IV | MPSTME NMIMS | "
    "Data: MoSPI eSankhyiki — CPI Base 2012=100 | All India Combined"
    "</div>", unsafe_allow_html=True
)
