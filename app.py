import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
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

# ── CSS — dark mode compatible ────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main header — white in dark mode, dark blue in light mode */
    .main-header {
        font-size: 2rem; font-weight: 700;
        text-align: center; padding: 0.8rem 0 0.2rem 0;
        color: #1a3a5c;
    }
    @media (prefers-color-scheme: dark) {
        .main-header { color: #ffffff; }
    }
    [data-theme="dark"] .main-header { color: #ffffff; }

    /* Sub-header */
    .sub-header {
        font-size: 0.95rem; color: #888;
        text-align: center; margin-bottom: 1rem;
    }

    /* Section titles — white in dark mode */
    .section-title {
        font-size: 1.35rem; font-weight: 600;
        border-left: 5px solid #e74c3c; padding-left: 10px; margin: 0.8rem 0;
        color: #1a3a5c;
    }
    @media (prefers-color-scheme: dark) {
        .section-title { color: #ffffff; }
    }
    [data-theme="dark"] .section-title { color: #ffffff !important; }

    /* Streamlit injects data-theme on the root — target all headings */
    html[data-theme="dark"] .section-title,
    .stApp[data-theme="dark"] .section-title {
        color: #ffffff !important;
    }

    /* Force white for section titles regardless of theme detection method */
    :root[data-theme="dark"] .section-title { color: #ffffff !important; }

    /* Reject / accept / info boxes — readable in both modes */
    .reject-box {
        background: rgba(231, 76, 60, 0.15); border-left: 4px solid #e74c3c;
        padding: 0.7rem 1rem; border-radius: 5px; margin: 0.5rem 0;
        color: inherit;
    }
    .accept-box {
        background: rgba(39, 174, 96, 0.15); border-left: 4px solid #27ae60;
        padding: 0.7rem 1rem; border-radius: 5px; margin: 0.5rem 0;
        color: inherit;
    }
    .info-box {
        background: rgba(41, 128, 185, 0.15); border-left: 4px solid #2980b9;
        padding: 0.7rem 1rem; border-radius: 5px; margin: 0.5rem 0;
        color: inherit;
    }
</style>

<script>
    // Inject data-theme onto .section-title elements based on Streamlit's theme
    const observer = new MutationObserver(() => {
        const isDark = document.body.classList.contains('dark') ||
            window.matchMedia('(prefers-color-scheme: dark)').matches ||
            document.documentElement.getAttribute('data-theme') === 'dark';
        document.querySelectorAll('.section-title').forEach(el => {
            el.style.color = isDark ? '#ffffff' : '#1a3a5c';
        });
        document.querySelectorAll('.main-header').forEach(el => {
            el.style.color = isDark ? '#ffffff' : '#1a3a5c';
        });
    });
    observer.observe(document.body, { attributes: true, childList: true, subtree: true });
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => observer.takeRecords());
</script>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
X_COLS   = ['Food_and_Beverages', 'Fuel_and_Light', 'Housing',
            'Clothing_and_Footwear', 'Miscellaneous']
X_LABELS = ['Food & Beverages', 'Fuel & Light', 'Housing',
            'Clothing & Footwear', 'Miscellaneous']
COLORS   = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

# ── Chart helper — transparent background works in both light & dark ──────────
def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor('none')
    ax.figure.patch.set_alpha(0.0)
    ax.tick_params(colors='#aaaaaa')
    ax.xaxis.label.set_color('#aaaaaa')
    ax.yaxis.label.set_color('#aaaaaa')
    ax.title.set_color('#dddddd')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')
    ax.grid(True, alpha=0.2, color='#666666')
    if title:   ax.set_title(title, fontsize=12, color='#dddddd')
    if xlabel:  ax.set_xlabel(xlabel, color='#aaaaaa')
    if ylabel:  ax.set_ylabel(ylabel, color='#aaaaaa')

def new_fig(w=12, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor('none')
    return fig, ax

def new_figs(rows, cols, w=12, h=4):
    fig, axes = plt.subplots(rows, cols, figsize=(w, h))
    fig.patch.set_alpha(0.0)
    if hasattr(axes, '__iter__'):
        for ax in np.array(axes).flatten():
            ax.set_facecolor('none')
    else:
        axes.set_facecolor('none')
    return fig, axes

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_excel("cpi_clean.xlsx")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['year', 'month_code']).reset_index(drop=True)
    # Build inflation rate columns (YoY pct change for each sub-group)
    for col in X_COLS:
        df[col + '_infl'] = df[col].pct_change(12) * 100
    df['General_infl'] = df['General'].pct_change(12) * 100
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
    'Statistical Analysis of India Consumer Price Index (2015–2025)'
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
    "🔮 CPI Forecasting 2026",
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

    st.markdown("#### 📉 General CPI Trend")
    fig, ax = new_fig(12, 3.5)
    ax.plot(dff['date'], dff['General'], color='#5b9bd5', linewidth=2, label='General CPI')
    if dff['date'].min() < pd.Timestamp('2020-01-01') < dff['date'].max():
        ax.axvline(pd.Timestamp('2020-01-01'), color='#e74c3c', linestyle='--',
                   linewidth=1.5, label='COVID Start (Jan 2020)')
        ax.fill_between(dff['date'], dff['General'].min(), dff['General'],
                        where=dff['date'] >= pd.Timestamp('2020-01-01'),
                        alpha=0.08, color='#e74c3c')
    ax.legend(fontsize=9, facecolor='none', labelcolor='#cccccc')
    style_ax(ax, 'All India General CPI — Monthly', 'Date', 'CPI Index (Base 2012=100)')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("#### 📈 CPI Sub-Group Trends")
    selected_groups = st.multiselect(
        "Select sub-groups to display", options=X_COLS, default=X_COLS,
        format_func=lambda x: x.replace('_', ' ')
    )
    if selected_groups:
        fig, ax = new_fig(12, 3.5)
        for col, color in zip(X_COLS, COLORS):
            if col in selected_groups:
                ax.plot(dff['date'], dff[col], label=col.replace('_', ' '),
                        color=color, linewidth=1.6)
        if dff['date'].min() < pd.Timestamp('2020-01-01') < dff['date'].max():
            ax.axvline(pd.Timestamp('2020-01-01'), color='#ffffff', linestyle='--',
                       linewidth=1.2, alpha=0.5, label='COVID Start')
        ax.legend(fontsize=8, loc='upper left', facecolor='none', labelcolor='#cccccc')
        style_ax(ax, 'CPI by Sub-Group — Monthly Trends', 'Date', 'CPI Index')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🔗 Correlation Matrix")
        fig, ax = new_fig(5.5, 4.5)
        sns.heatmap(dff[X_COLS + ['General']].corr(), annot=True, cmap='coolwarm',
                    fmt='.2f', ax=ax, linewidths=0.5, square=True,
                    vmin=-1, vmax=1,
                    annot_kws={"size": 9},
                    cbar_kws={"shrink": 0.8})
        ax.set_title('Correlation Matrix', fontsize=11, color='#dddddd')
        ax.tick_params(colors='#aaaaaa', labelsize=8)
        fig.patch.set_alpha(0.0); ax.set_facecolor('none')
        plt.tight_layout(); st.pyplot(fig); plt.close()

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
        "Z-Test: Large sample — Pre vs Post COVID"
    ])
    alpha = st.selectbox("Significance Level (α)", [0.01, 0.05, 0.10], index=1)
    tail  = st.radio("Tail", ["Two-tailed", "One-tailed (right)", "One-tailed (left)"],
                     horizontal=True)

    pre_data  = df[df['covid_period'] == 'Pre-COVID']['General'].values
    post_data = df[df['covid_period'] == 'Post-COVID']['General'].values
    st.markdown("---")

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
            t_crit = t.ppf(1 - alpha/2, df_t); p_val = 2*(1 - t.cdf(abs(t_cal), df_t))
        elif tail == "One-tailed (right)":
            t_crit = t.ppf(1 - alpha, df_t);   p_val = 1 - t.cdf(t_cal, df_t)
        else:
            t_crit = t.ppf(alpha, df_t);        p_val = t.cdf(t_cal, df_t)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pre-COVID Mean",  f"{xb2:.2f}")
        c2.metric("Post-COVID Mean", f"{xb1:.2f}")
        c3.metric("t-calculated",    f"{t_cal:.4f}")
        c4.metric("p-value",         f"{p_val:.6f}")
        st.metric("t-critical", f"±{t_crit:.4f}" if tail == "Two-tailed" else f"{t_crit:.4f}")

        if p_val < alpha:
            st.markdown(f'<div class="reject-box">🔴 <b>Reject H₀</b> — Post-COVID CPI is significantly higher (p = {p_val:.4f} &lt; α = {alpha})</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="accept-box">🟢 <b>Fail to Reject H₀</b> — No significant difference (p = {p_val:.4f} ≥ α = {alpha})</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = new_fig(6, 4)
            x_r = np.linspace(-5, 25, 400)
            ax.plot(x_r, t.pdf(x_r, df_t), '#5b9bd5', linewidth=2, label='t-distribution')
            ax.axvline(t_cal,  color='#e74c3c', linestyle='--', linewidth=2, label=f't-cal = {t_cal:.3f}')
            ax.axvline(abs(t_crit), color='#2ecc71', linestyle='--', linewidth=2, label=f't-crit = {abs(t_crit):.3f}')
            if tail == "One-tailed (right)":
                xf = np.linspace(t_crit, 25, 100)
                ax.fill_between(xf, t.pdf(xf, df_t), alpha=0.3, color='#e74c3c', label='Rejection Region')
            elif tail == "One-tailed (left)":
                xf = np.linspace(-5, t_crit, 100)
                ax.fill_between(xf, t.pdf(xf, df_t), alpha=0.3, color='#e74c3c', label='Rejection Region')
            else:
                ax.fill_between(np.linspace(-5, -abs(t_crit), 100),
                                t.pdf(np.linspace(-5, -abs(t_crit), 100), df_t), alpha=0.3, color='#e74c3c')
                xf = np.linspace(abs(t_crit), 25, 100)
                ax.fill_between(xf, t.pdf(xf, df_t), alpha=0.3, color='#e74c3c', label='Rejection Region')
            ax.legend(fontsize=8, facecolor='none', labelcolor='#cccccc')
            style_ax(ax, f't-Distribution | df={df_t}', 't value', 'Density')
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col2:
            fig, ax = new_fig(6, 4)
            bp = ax.boxplot([pre_data, post_data], labels=['Pre-COVID', 'Post-COVID'],
                            patch_artist=True,
                            boxprops=dict(facecolor='#5b9bd5', alpha=0.5),
                            medianprops=dict(color='#e74c3c', linewidth=2),
                            whiskerprops=dict(color='#aaaaaa'),
                            capprops=dict(color='#aaaaaa'),
                            flierprops=dict(markerfacecolor='#aaaaaa'))
            style_ax(ax, 'CPI Distribution: Pre vs Post COVID', '', 'General CPI Index')
            plt.tight_layout(); st.pyplot(fig); plt.close()

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
            z_crit = norm.ppf(1 - alpha/2); p_val = 2*(1 - norm.cdf(abs(z_cal)))
        elif tail == "One-tailed (right)":
            z_crit = norm.ppf(1 - alpha);   p_val = 1 - norm.cdf(z_cal)
        else:
            z_crit = norm.ppf(alpha);        p_val = norm.cdf(z_cal)

        c1, c2, c3 = st.columns(3)
        c1.metric("z-calculated", f"{z_cal:.4f}")
        c2.metric("z-critical",   f"{z_crit:.4f}")
        c3.metric("p-value",      f"{p_val:.6f}")

        if p_val < alpha:
            st.markdown(f'<div class="reject-box">🔴 <b>Reject H₀</b> — Significant difference (p = {p_val:.4f} &lt; α = {alpha})</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="accept-box">🟢 <b>Fail to Reject H₀</b></div>', unsafe_allow_html=True)

        fig, ax = new_fig(10, 4)
        x_r = np.linspace(-4, 25, 400)
        ax.plot(x_r, norm.pdf(x_r), '#5b9bd5', linewidth=2, label='Standard Normal')
        ax.axvline(z_cal,  color='#e74c3c', linestyle='--', linewidth=2, label=f'z-cal = {z_cal:.3f}')
        ax.axvline(z_crit, color='#2ecc71', linestyle='--', linewidth=2, label=f'z-crit = {z_crit:.3f}')
        if tail == "One-tailed (right)":
            xf = np.linspace(z_crit, 25, 100)
            ax.fill_between(xf, norm.pdf(xf), alpha=0.3, color='#e74c3c', label='Rejection Region')
        ax.legend(fontsize=9, facecolor='none', labelcolor='#cccccc')
        style_ax(ax, f'Z-Test Distribution | α={alpha}', 'z value', 'Density')
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 3 — ANOVA
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "📊 ANOVA":
    st.markdown('<div class="section-title">ANOVA — Analysis of Variance</div>',
                unsafe_allow_html=True)

    anova_type = st.radio("Select ANOVA Type",
                          ["One-Way ANOVA", "Two-Way ANOVA (with Interaction)", "Tukey HSD Post-Hoc Test"],
                          horizontal=True)
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
            st.markdown(f'<div class="reject-box">🔴 <b>Reject H₀</b> — Significant differences between group means (p &lt; {alpha_a})</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="accept-box">🟢 <b>Fail to Reject H₀</b></div>', unsafe_allow_html=True)

        st.markdown("**ANOVA Table (statsmodels):**")
        st.dataframe(anova_tbl.round(4), use_container_width=True)

        fig, ax = new_fig(10, 4)
        sns.boxplot(data=df_melt, x='Group', y='CPI_Index', palette='Set2', ax=ax)
        style_ax(ax, f'CPI by Sub-Group  |  F = {f_stat:.2f},  p = {p_val:.4f}', '', 'CPI Index')
        plt.xticks(rotation=12, color='#aaaaaa')
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
            st.markdown(f'<div class="reject-box">🔴 Significant effects found for: <b>{", ".join(sig)}</b></div>', unsafe_allow_html=True)

        fig, ax = new_fig(11, 4)
        sns.boxplot(data=df_melt, x='Group', y='CPI_Index', hue='covid_period',
                    palette={'Pre-COVID': '#5b9bd5', 'Post-COVID': '#e74c3c'}, ax=ax)
        style_ax(ax, 'CPI by Sub-Group × COVID Period', '', 'CPI Index')
        plt.xticks(rotation=12, color='#aaaaaa')
        ax.legend(title='Period', fontsize=9, facecolor='none', labelcolor='#cccccc')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    else:
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
                lambda row: ['background-color:rgba(231,76,60,0.2)' if row['reject'] else '' for _ in row],
                axis=1),
            use_container_width=True
        )
        group_means = df_melt.groupby('Group')['CPI_Index'].mean().sort_values()
        fig, ax = new_fig(8, 4)
        ax.barh(group_means.index, group_means.values, color='#5b9bd5', alpha=0.8)
        style_ax(ax, 'Mean CPI by Sub-Group', 'Mean CPI Index', '')
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 4 — OLS REGRESSION — now uses YoY inflation rates (non-circular)
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "📈 OLS Linear Regression":
    st.markdown('<div class="section-title">Multiple Linear Regression (OLS) — Inflation Rates</div>',
                unsafe_allow_html=True)
    st.markdown("""
    **Target variable (Y):** Year-over-Year General CPI Inflation Rate (%)  
    **Predictors (X):** Year-over-Year Inflation Rates of each sub-group (%)  
    This version predicts **how fast prices are rising**, not the index level itself — making it a genuine, non-circular predictive model.
    """)

    INFL_COLS = [c + '_infl' for c in X_COLS]
    INFL_LABELS = [l + ' Infl%' for l in X_LABELS]

    selected_x = st.multiselect(
        "Select predictor variables (X)", options=INFL_COLS, default=INFL_COLS,
        format_func=lambda x: x.replace('_infl', '').replace('_', ' ') + ' Inflation Rate'
    )
    if not selected_x:
        st.warning("Please select at least one predictor."); st.stop()

    df_reg = df[selected_x + ['General_infl']].dropna()
    formula = 'General_infl ~ ' + ' + '.join(selected_x)
    fit = smf.ols(formula, data=df_reg).fit()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R²",          f"{fit.rsquared:.4f}")
    c2.metric("Adj. R²",     f"{fit.rsquared_adj:.4f}")
    c3.metric("F-statistic", f"{fit.fvalue:.2f}")
    c4.metric("F p-value",   f"{fit.f_pvalue:.6f}")

    if fit.rsquared >= 0.99:
        st.markdown('<div class="info-box">ℹ️ R² is very high because sub-group inflation rates are direct components of General inflation. The coefficient values and significance tests are still meaningful.</div>', unsafe_allow_html=True)

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
            'Feature': [s.replace('_infl','').replace('_',' ') for s in selected_x],
            'VIF':     [variance_inflation_factor(X_vif.values, i) for i in range(len(selected_x))]
        }).round(2)
        st.dataframe(vif_df, use_container_width=True)
        max_vif = vif_df['VIF'].max()
        if max_vif > 10:
            st.markdown(f'<div class="reject-box">⚠️ High VIF ({max_vif:.1f}) detected — multicollinearity present. Ridge &amp; Lasso can address this.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="accept-box">✅ VIF values are acceptable (max {max_vif:.1f} &lt; 10) — multicollinearity is not a concern here.</div>', unsafe_allow_html=True)

    # Diagnostic plots
    fitted    = fit.fittedvalues
    residuals = fit.resid
    fig, axes = new_figs(1, 2, 11, 4)
    axes[0].scatter(fitted, residuals, alpha=0.5, color='#5b9bd5', s=25)
    axes[0].axhline(0, color='#e74c3c', linestyle='--', linewidth=1.5)
    style_ax(axes[0], 'Residuals vs Fitted Values', 'Fitted Values', 'Residuals')
    sm.qqplot(residuals, line='s', ax=axes[1], alpha=0.5)
    axes[1].get_lines()[0].set_color('#5b9bd5')
    style_ax(axes[1], 'Q-Q Plot of Residuals')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("#### Actual vs Predicted YoY Inflation Rate")
    fig, ax = new_fig(12, 3.5)
    ax.plot(df_reg.index, df_reg['General_infl'], label='Actual',
            color='#5b9bd5', linewidth=1.8)
    ax.plot(df_reg.index, fitted, label='Predicted',
            color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.85)
    ax.axhline(6, color='#f39c12', linestyle=':', linewidth=1.2, label='RBI Upper Limit (6%)')
    ax.axhline(2, color='#2ecc71', linestyle=':', linewidth=1.2, label='RBI Lower Limit (2%)')
    ax.legend(fontsize=8, facecolor='none', labelcolor='#cccccc')
    style_ax(ax, 'Actual vs Predicted General CPI YoY Inflation Rate (%)',
             'Observation Index', 'Inflation Rate (%)')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("#### Full OLS Summary")
    st.text(str(fit.summary()))

# ═══════════════════════════════════════════════════════════════════════════════
# 5 — RIDGE & LASSO — also on inflation rates
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "🔵 Ridge & Lasso Regression":
    st.markdown('<div class="section-title">Ridge & Lasso Regression — Inflation Rates</div>',
                unsafe_allow_html=True)
    st.markdown("Regularization on YoY inflation rates — coefficient paths + KFold CV. Lasso will zero out weak predictors, giving genuine feature selection.")

    INFL_COLS = [c + '_infl' for c in X_COLS]
    k_folds = st.slider("KFold splits (K)", 3, 15, 10)

    df_reg = df[INFL_COLS + ['General_infl']].dropna()
    X      = df_reg[INFL_COLS].values
    Y      = df_reg['General_infl'].values
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    alphas = 10**np.linspace(4, -3, 100)

    ridge_coefs, lasso_coefs = [], []
    for a in alphas:
        r = Ridge(alpha=a); r.fit(X_sc, Y); ridge_coefs.append(r.coef_)
        try:
            l = Lasso(alpha=a, max_iter=20000); l.fit(X_sc, Y); lasso_coefs.append(l.coef_)
        except Exception:
            lasso_coefs.append([0] * len(INFL_COLS))

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = new_fig(6, 4)
        for i, (label, color) in enumerate(zip(X_LABELS, COLORS)):
            ax.plot(alphas, [c[i] for c in ridge_coefs], color=color, label=label, linewidth=1.6)
        ax.set_xscale('log')
        ax.legend(fontsize=7, loc='upper right', facecolor='none', labelcolor='#cccccc')
        style_ax(ax, 'Ridge — Coefficient Path', 'Alpha (λ)', 'Coefficient')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        fig, ax = new_fig(6, 4)
        for i, (label, color) in enumerate(zip(X_LABELS, COLORS)):
            ax.plot(alphas, [c[i] for c in lasso_coefs], color=color, label=label, linewidth=1.6)
        ax.set_xscale('log')
        ax.legend(fontsize=7, loc='upper right', facecolor='none', labelcolor='#cccccc')
        style_ax(ax, 'Lasso — Coefficient Path (Features Zeroed Out)', 'Alpha (λ)', 'Coefficient')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown(f"---\n#### Cross-Validation (KFold = {k_folds}) — Best Alpha Selection")
    kfold   = KFold(k_folds, random_state=0, shuffle=True)
    ridgeCV = RidgeCV(alphas=alphas, cv=kfold); ridgeCV.fit(X_sc, Y)
    lassoCV = LassoCV(alphas=alphas, cv=kfold, max_iter=20000); lassoCV.fit(X_sc, Y)
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
        st.metric("Best Alpha", f"{ridgeCV.alpha_:.6f}")
        st.metric("R²",         f"{ridgeCV.score(X_sc, Y):.4f}")
        st.metric("MSE",        f"{ridge_mse:.4f}")
    with c3:
        st.markdown("**📌 Lasso**")
        st.metric("Best Alpha", f"{lassoCV.alpha_:.6f}")
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

    fig, ax = new_fig(10, 4)
    x_pos = np.arange(len(X_LABELS))
    ax.bar(x_pos - 0.25, lin_model.coef_, 0.25, label='OLS',   color='#95a5a6')
    ax.bar(x_pos,        ridgeCV.coef_,   0.25, label='Ridge', color='#5b9bd5')
    ax.bar(x_pos + 0.25, lassoCV.coef_,   0.25, label='Lasso', color='#e74c3c')
    ax.set_xticks(x_pos); ax.set_xticklabels(X_LABELS, rotation=12, color='#aaaaaa')
    ax.axhline(0, color='#aaaaaa', linewidth=0.8)
    ax.legend(facecolor='none', labelcolor='#cccccc')
    style_ax(ax, 'OLS vs Ridge vs Lasso — Coefficient Comparison (Standardized)',
             '', 'Coefficient')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    zeroed = [X_LABELS[i] for i, c in enumerate(lassoCV.coef_) if abs(c) < 0.001]
    if zeroed:
        st.markdown(f'<div class="accept-box">✅ <b>Lasso zeroed out:</b> {", ".join(zeroed)} — these sub-groups do not independently predict General inflation rate</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">ℹ️ Lasso retained all features. All sub-group inflation rates contribute to predicting General inflation.</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 6 — FORECASTING 2026
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "🔮 CPI Forecasting 2026":
    st.markdown('<div class="section-title">CPI Forecasting — 2026 Projections</div>',
                unsafe_allow_html=True)
    st.markdown("""
    Using **OLS time-trend regression** to forecast General CPI and all sub-group indices 
    for the 12 months of 2026. The model fits a polynomial trend to historical data 
    and extrapolates forward. Shaded bands show the **95% confidence interval**.
    """)

    conf_level = st.slider("Confidence Interval Level (%)", 80, 99, 95)
    show_subgroups = st.multiselect(
        "Select sub-groups to forecast",
        options=X_COLS, default=X_COLS,
        format_func=lambda x: x.replace('_', ' ')
    )

    # Build time index
    df_fc = df.copy()
    df_fc['t'] = np.arange(len(df_fc))
    df_fc['t2'] = df_fc['t'] ** 2

    # Future months: Jan-Dec 2026
    last_t = df_fc['t'].max()
    future_dates = pd.date_range('2026-01-01', periods=12, freq='MS')
    future_t  = np.arange(last_t + 1, last_t + 13)
    future_df = pd.DataFrame({'t': future_t, 't2': future_t**2})

    alpha_fc = 1 - conf_level / 100

    def forecast_series(target_col):
        fit = smf.ols(f'{target_col} ~ t + t2', data=df_fc).fit()
        pred = fit.get_prediction(future_df)
        mean  = pred.predicted_mean
        ci    = pred.conf_int(alpha=alpha_fc)
        return mean, ci[:, 0], ci[:, 1]

    st.markdown("---")
    st.markdown("#### 📈 General CPI Forecast — 2026")
    gen_mean, gen_lo, gen_hi = forecast_series('General')

    fig, ax = new_fig(13, 4)
    ax.plot(df_fc['date'], df_fc['General'],
            color='#5b9bd5', linewidth=2, label='Historical General CPI')
    ax.plot(future_dates, gen_mean,
            color='#f39c12', linewidth=2.5, linestyle='--', label='Forecast 2026')
    ax.fill_between(future_dates, gen_lo, gen_hi,
                    alpha=0.25, color='#f39c12', label=f'{conf_level}% CI')
    ax.axvline(pd.Timestamp('2026-01-01'), color='#aaaaaa', linestyle=':', linewidth=1.2)
    ax.axvline(pd.Timestamp('2020-01-01'), color='#e74c3c', linestyle='--',
               linewidth=1.2, alpha=0.6, label='COVID Start')
    ax.legend(fontsize=9, facecolor='none', labelcolor='#cccccc')
    style_ax(ax, 'General CPI — Historical & 2026 Forecast', 'Date', 'CPI Index')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Forecast table
    fc_table = pd.DataFrame({
        'Month': future_dates.strftime('%b 2026'),
        'Forecasted General CPI': gen_mean.round(2),
        f'Lower {conf_level}% CI': gen_lo.round(2),
        f'Upper {conf_level}% CI': gen_hi.round(2),
    })
    st.dataframe(fc_table, use_container_width=True)

    last_actual = df_fc['General'].iloc[-1]
    expected_change = gen_mean[-1] - last_actual
    st.markdown(f'<div class="info-box">📌 <b>Forecast summary:</b> General CPI is projected to rise from <b>{last_actual:.1f}</b> (Dec 2025) to approximately <b>{gen_mean[-1]:.1f}</b> by Dec 2026 — an increase of <b>{expected_change:.1f} points ({expected_change/last_actual*100:.1f}%)</b> over the year.</div>', unsafe_allow_html=True)

    if show_subgroups:
        st.markdown("---")
        st.markdown("#### 📊 Sub-Group CPI Forecasts — 2026")

        n_cols = 2
        n_rows = (len(show_subgroups) + 1) // 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 4 * n_rows))
        fig.patch.set_alpha(0.0)
        axes_flat = np.array(axes).flatten() if n_rows > 1 else [axes[0], axes[1]]

        for idx, col in enumerate(show_subgroups):
            ax = axes_flat[idx]
            ax.set_facecolor('none')
            color = COLORS[X_COLS.index(col)]
            sg_mean, sg_lo, sg_hi = forecast_series(col)
            ax.plot(df_fc['date'], df_fc[col],
                    color=color, linewidth=1.8, alpha=0.8, label='Historical')
            ax.plot(future_dates, sg_mean,
                    color='#f39c12', linewidth=2, linestyle='--', label='Forecast')
            ax.fill_between(future_dates, sg_lo, sg_hi,
                            alpha=0.2, color='#f39c12', label=f'{conf_level}% CI')
            ax.axvline(pd.Timestamp('2026-01-01'), color='#aaaaaa', linestyle=':', linewidth=1)
            style_ax(ax, col.replace('_', ' '), 'Date', 'CPI Index')
            ax.legend(fontsize=7, facecolor='none', labelcolor='#cccccc')

        # Hide empty subplots
        for idx in range(len(show_subgroups), len(axes_flat)):
            axes_flat[idx].set_visible(False)

        plt.tight_layout(); st.pyplot(fig); plt.close()

        # Sub-group forecast summary table
        st.markdown("#### Forecast Summary — Dec 2026 Projections")
        summary_rows = []
        for col in show_subgroups:
            sg_mean, sg_lo, sg_hi = forecast_series(col)
            last_val = df_fc[col].iloc[-1]
            summary_rows.append({
                'Sub-Group':       col.replace('_', ' '),
                'Dec 2025 (Actual)': f"{last_val:.1f}",
                'Dec 2026 (Forecast)': f"{sg_mean[-1]:.1f}",
                'Change':          f"+{sg_mean[-1]-last_val:.1f}",
                f'Lower {conf_level}% CI': f"{sg_lo[-1]:.1f}",
                f'Upper {conf_level}% CI': f"{sg_hi[-1]:.1f}",
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    st.markdown('<div class="info-box">⚠️ <b>Limitation:</b> This forecast uses a simple polynomial time-trend OLS model. It does not account for external shocks (oil price changes, monsoon failures, policy changes). Actual 2026 CPI may differ. A time-series model like ARIMA would be more rigorous for production forecasting.</div>', unsafe_allow_html=True)
