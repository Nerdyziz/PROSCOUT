"""
=============================================================================
 ProScout AI: Professional Footballer Market Valuation & Scouting Studio
 Modern Sports Analytics & Transfer Intelligence Platform (2022 Dataset)
=============================================================================
"""

import os
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import joblib

# Page configuration
st.set_page_config(
    page_title="ProScout | Footballer Valuation Studio",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Athletic & Modern UI Styling via st.html
st.html(textwrap.dedent("""
<style>
    /* Global Background */
    .stApp {
        background-color: #0B0F19;
        color: #F3F4F6;
    }
    
    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F172A 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 22px 28px;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 900;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #38BDF8, #34D399, #FBBF24);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        padding: 0;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #94A3B8;
        margin-top: 5px;
    }
    
    /* FUT Card Styling */
    .fut-card {
        background: linear-gradient(160deg, #1E293B 0%, #0F172A 100%);
        border: 2px solid #F59E0B;
        border-radius: 20px;
        padding: 22px;
        box-shadow: 0 12px 35px -8px rgba(245, 158, 11, 0.3);
        text-align: center;
        margin-bottom: 12px;
    }
    .fut-ovr {
        font-size: 3.2rem;
        font-weight: 900;
        line-height: 1;
        color: #FBBF24;
        text-shadow: 0 0 18px rgba(251, 191, 36, 0.4);
    }
    .fut-pos-badge {
        background: #10B981;
        color: #064E3B;
        font-weight: 800;
        font-size: 0.9rem;
        padding: 3px 12px;
        border-radius: 8px;
        display: inline-block;
        margin-top: 4px;
    }
    .fut-name {
        font-size: 1.65rem;
        font-weight: 800;
        color: #F8FAFC;
        margin: 12px 0 4px 0;
        letter-spacing: 0.5px;
    }
    .valuation-box {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 14px;
        padding: 14px;
        margin: 14px 0;
    }
    .val-amount {
        font-size: 2.3rem;
        font-weight: 900;
        color: #34D399;
        letter-spacing: -0.5px;
    }
    .val-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #94A3B8;
        font-weight: 600;
    }
    
    /* Stat Pills */
    .stat-pill {
        background: #1F2937;
        border-radius: 10px;
        padding: 8px 10px;
        text-align: center;
        border: 1px solid #374151;
    }
    .stat-pill-label {
        font-size: 0.72rem;
        font-weight: 700;
        color: #9CA3AF;
        text-transform: uppercase;
    }
    .stat-pill-val {
        font-size: 1.25rem;
        font-weight: 800;
        color: #F9FAFB;
    }
    
    /* Twin Cards */
    .twin-card {
        background: #131D31;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 10px;
    }
    
    /* Metric KPI Cards */
    .kpi-card {
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .kpi-num {
        font-size: 1.8rem;
        font-weight: 800;
        color: #38BDF8;
    }
    .kpi-lbl {
        font-size: 0.8rem;
        color: #9CA3AF;
        text-transform: uppercase;
        margin-top: 4px;
    }
</style>
"""))

# -----------------------------------------------------------------------------
# 1. Load ML Artifacts
# -----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model_path = "evaluation_results/saved_models/best_footballer_valuation_model.joblib"
    prep_path = "evaluation_results/saved_models/football_preprocessor.joblib"
    engine_path = "evaluation_results/saved_models/scout_similarity_engine.joblib"
    
    if not (os.path.exists(model_path) and os.path.exists(prep_path) and os.path.exists(engine_path)):
        st.error("Model artifacts not found! Run training pipeline first.")
        st.stop()
        
    model = joblib.load(model_path)
    preprocessor = joblib.load(prep_path)
    engine_data = joblib.load(engine_path)
    return model, preprocessor, engine_data

model, preprocessor, engine_data = load_artifacts()
nn_model = engine_data['nn_model']
scout_scaler = engine_data['scout_scaler']
player_db = engine_data['player_db']
scout_cols = engine_data['scout_cols']

# -----------------------------------------------------------------------------
# 2. Player Presets (Updated to 2022 Season)
# -----------------------------------------------------------------------------
PRESETS = {
    "⚡ Kylian Mbappé": {
        "name": "Kylian Mbappé", "pos": "Forward", "foot": "Right", "age": 22,
        "overall": 91, "potential": 95, "rep": 4, "wf": 4, "sm": 5,
        "pace": 97, "shooting": 88, "passing": 80, "dribbling": 92, "defending": 36, "physic": 77
    },
    "🤖 Erling Haaland": {
        "name": "Erling Haaland", "pos": "Forward", "foot": "Left", "age": 20,
        "overall": 88, "potential": 93, "rep": 4, "wf": 3, "sm": 3,
        "pace": 89, "shooting": 91, "passing": 65, "dribbling": 80, "defending": 45, "physic": 88
    },
    "👑 Lionel Messi": {
        "name": "Lionel Messi", "pos": "Forward", "foot": "Left", "age": 34,
        "overall": 93, "potential": 93, "rep": 5, "wf": 4, "sm": 4,
        "pace": 85, "shooting": 92, "passing": 91, "dribbling": 95, "defending": 34, "physic": 65
    },
    "🎯 Robert Lewandowski": {
        "name": "Robert Lewandowski", "pos": "Forward", "foot": "Right", "age": 32,
        "overall": 92, "potential": 92, "rep": 5, "wf": 4, "sm": 4,
        "pace": 78, "shooting": 92, "passing": 79, "dribbling": 86, "defending": 44, "physic": 82
    },
    "🚀 Cristiano Ronaldo": {
        "name": "Cristiano Ronaldo", "pos": "Forward", "foot": "Right", "age": 36,
        "overall": 91, "potential": 91, "rep": 5, "wf": 4, "sm": 5,
        "pace": 87, "shooting": 94, "passing": 80, "dribbling": 88, "defending": 34, "physic": 75
    },
    "🪄 Kevin De Bruyne": {
        "name": "Kevin De Bruyne", "pos": "Midfielder", "foot": "Right", "age": 30,
        "overall": 91, "potential": 91, "rep": 4, "wf": 5, "sm": 4,
        "pace": 76, "shooting": 86, "passing": 93, "dribbling": 88, "defending": 64, "physic": 78
    },
    "💎 Pedri (Wonderkid)": {
        "name": "Pedri", "pos": "Midfielder", "foot": "Right", "age": 18,
        "overall": 81, "potential": 91, "rep": 2, "wf": 4, "sm": 4,
        "pace": 78, "shooting": 65, "passing": 79, "dribbling": 84, "defending": 66, "physic": 63
    },
    "🧤 Manuel Neuer": {
        "name": "Manuel Neuer", "pos": "Goalkeeper", "foot": "Right", "age": 35,
        "overall": 90, "potential": 90, "rep": 5, "wf": 4, "sm": 1,
        "pace": 61, "shooting": 42, "passing": 41, "dribbling": 31, "defending": 25, "physic": 83
    },
    "⚙️ Custom Athlete (Build from Scratch)": {
        "name": "Custom Player", "pos": "Midfielder", "foot": "Right", "age": 24,
        "overall": 80, "potential": 85, "rep": 2, "wf": 3, "sm": 3,
        "pace": 78, "shooting": 74, "passing": 80, "dribbling": 81, "defending": 65, "physic": 72
    }
}

# -----------------------------------------------------------------------------
# 3. Hero Header
# -----------------------------------------------------------------------------
st.html(textwrap.dedent("""
<div class="hero-banner">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <h1 class="hero-title">⚽ PROSCOUT AI</h1>
            <p class="hero-subtitle">Transfer Market Valuation & Tactical Player Intelligence Studio (FIFA 2022)</p>
        </div>
        <div style="display: flex; gap: 12px; margin-top: 10px;">
            <div style="background: rgba(56, 189, 248, 0.1); border: 1px solid #0284C7; padding: 6px 14px; border-radius: 20px; font-size: 0.85rem; color: #38BDF8; font-weight: 700;">
                ✓ 19,165 Verified Athletes (2022)
            </div>
            <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid #059669; padding: 6px 14px; border-radius: 20px; font-size: 0.85rem; color: #34D399; font-weight: 700;">
                ✓ 99.7% Accuracy (±20%)
            </div>
            <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid #D97706; padding: 6px 14px; border-radius: 20px; font-size: 0.85rem; color: #FBBF24; font-weight: 700;">
                ✓ 23 ML Attributes
            </div>
        </div>
    </div>
</div>
"""))

# Main Navigation Tabs
tab_studio, tab_analytics = st.tabs(["🎯 Player Valuation & Scout Studio", "📊 Model Performance & Insights"])

# =============================================================================
# TAB 1: VALUATION & SCOUT STUDIO
# =============================================================================
with tab_studio:
    # Quick Preset Bar
    st.markdown("##### ⚡ Quick Load Player Profile (2022 Roster)")
    preset_choice = st.selectbox(
        "Choose an existing star profile or customize below:",
        options=list(PRESETS.keys()),
        index=0,
        label_visibility="collapsed"
    )
    p = PRESETS[preset_choice]

    col_config, col_card = st.columns([1.1, 0.9], gap="large")

    # -------------------------------------------------------------------------
    # LEFT COLUMN: Configuration
    # -------------------------------------------------------------------------
    with col_config:
        st.markdown("#### 1. Identity & Core Profile")
        c1, c2, c3 = st.columns([1.5, 1, 1])
        with c1:
            player_name = st.text_input("Player Name", value=p["name"])
        with c2:
            position_group = st.selectbox(
                "Position Group",
                ["Forward", "Midfielder", "Defender", "Goalkeeper"],
                index=["Forward", "Midfielder", "Defender", "Goalkeeper"].index(p["pos"])
            )
        with c3:
            preferred_foot = st.selectbox(
                "Preferred Foot",
                ["Right", "Left"],
                index=["Right", "Left"].index(p["foot"])
            )

        c4, c5, c6 = st.columns(3)
        with c4:
            age = st.slider("Age", min_value=16, max_value=42, value=p["age"])
        with c5:
            overall = st.slider("Overall Rating", min_value=50, max_value=95, value=p["overall"])
        with c6:
            potential = st.slider("Potential Rating", min_value=overall, max_value=96, value=max(overall, p["potential"]))

        c7, c8, c9 = st.columns(3)
        with c7:
            reputation = st.selectbox("International Rep", [1, 2, 3, 4, 5], index=p["rep"] - 1, format_func=lambda x: f"{'⭐' * x} ({x})")
        with c8:
            weak_foot = st.selectbox("Weak Foot Ability", [1, 2, 3, 4, 5], index=p["wf"] - 1, format_func=lambda x: f"{x} Stars")
        with c9:
            skill_moves = st.selectbox("Skill Moves", [1, 2, 3, 4, 5], index=p["sm"] - 1, format_func=lambda x: f"{x} Stars")

        st.markdown("---")
        st.markdown("#### 2. Tactical & Physical Attributes (FIFA Ratings)")
        
        tab_atk, tab_def = st.tabs(["⚽ Attacking & Playmaking", "🛡️ Defending & Physicality"])
        with tab_atk:
            ca1, ca2 = st.columns(2)
            with ca1:
                pace = st.slider("Pace (Sprint & Acceleration)", 30, 99, int(p["pace"]))
                shooting = st.slider("Shooting (Finishing & Power)", 30, 99, int(p["shooting"]))
            with ca2:
                passing = st.slider("Passing (Vision & Crossing)", 30, 99, int(p["passing"]))
                dribbling = st.slider("Dribbling (Agility & Control)", 30, 99, int(p["dribbling"]))
        with tab_def:
            cd1, cd2 = st.columns(2)
            with cd1:
                defending = st.slider("Defending (Tackles & Interceptions)", 20, 99, int(p["defending"]))
            with cd2:
                physic = st.slider("Physicality (Stamina & Strength)", 30, 99, int(p["physic"]))

    # -------------------------------------------------------------------------
    # VALUATION CALCULATION
    # -------------------------------------------------------------------------
    potential_growth = max(0, potential - overall)
    is_in_prime_age = 1 if (24 <= age <= 29) else 0
    overall_squared = (overall / 10.0) ** 2
    technical_rating = (shooting + passing + dribbling) / 3.0
    physical_rating = (pace + physic) / 2.0

    input_df = pd.DataFrame([{
        'age': age,
        'overall': overall,
        'potential': potential,
        'potential_growth': potential_growth,
        'international_reputation': reputation,
        'weak_foot': weak_foot,
        'skill_moves': skill_moves,
        'pace': float(pace),
        'shooting': float(shooting),
        'passing': float(passing),
        'dribbling': float(dribbling),
        'defending': float(defending),
        'physic': float(physic),
        'technical_rating': technical_rating,
        'physical_rating': physical_rating,
        'is_in_prime_age': is_in_prime_age,
        'overall_squared': overall_squared,
        'position_group': position_group,
        'preferred_foot': preferred_foot
    }])

    # Transform & Predict
    X_proc = preprocessor.transform(input_df)
    pred_log = model.predict(X_proc)[0]
    pred_eur = float(np.exp(pred_log))

    lower_eur = pred_eur * 0.85
    upper_eur = pred_eur * 1.15
    est_wage_eur = max(1000.0, pred_eur * 0.0025)

    # Market Tier
    if overall >= 90:
        tier_title = "🔥 GLOBAL SUPERSTAR"
        tier_color = "#F59E0B"
    elif age <= 21 and potential >= 86:
        tier_title = "⭐ ELITE WONDERKID"
        tier_color = "#38BDF8"
    elif overall >= 83:
        tier_title = "💎 PRIME STARTER"
        tier_color = "#34D399"
    elif age <= 23 and potential_growth >= 6:
        tier_title = "🌱 RISING PROSPECT"
        tier_color = "#A78BFA"
    else:
        tier_title = "🛡️ SQUAD PLAYER"
        tier_color = "#9CA3AF"

    def format_money(val):
        if val >= 1_000_000:
            return f"€{val / 1e6:.1f}M"
        return f"€{val / 1e3:.0f}K"

    # -------------------------------------------------------------------------
    # RIGHT COLUMN: Clean FUT Card rendered via st.html
    # -------------------------------------------------------------------------
    with col_card:
        card_html = f"""
        <div class="fut-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div style="text-align: left;">
                    <div class="fut-ovr">{overall}</div>
                    <div class="fut-pos-badge">{position_group.upper()}</div>
                    <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 4px;">{preferred_foot[0]} · Age {age}</div>
                </div>
                <div style="text-align: right;">
                    <div style="background: rgba(245, 158, 11, 0.15); border: 1px solid {tier_color}; color: {tier_color}; padding: 4px 10px; border-radius: 12px; font-weight: 800; font-size: 0.75rem;">
                        {tier_title}
                    </div>
                    <div style="color: #FBBF24; font-size: 0.85rem; margin-top: 6px;">{'⭐' * reputation}</div>
                </div>
            </div>
            
            <div class="fut-name">{player_name}</div>
            
            <div class="valuation-box">
                <div class="val-label">ESTIMATED MARKET VALUATION</div>
                <div class="val-amount">{format_money(pred_eur)}</div>
                <div style="color: #94A3B8; font-size: 0.82rem; margin-top: 4px;">
                    Fair Transfer Window: <strong style="color: #E2E8F0;">{format_money(lower_eur)} — {format_money(upper_eur)}</strong>
                </div>
                <div style="color: #38BDF8; font-size: 0.82rem; margin-top: 3px;">
                    Expected Wage: <strong style="color: #E2E8F0;">{format_money(est_wage_eur)} / week</strong>
                </div>
            </div>

            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 10px;">
                <div class="stat-pill"><div class="stat-pill-label">PAC</div><div class="stat-pill-val">{pace}</div></div>
                <div class="stat-pill"><div class="stat-pill-label">SHO</div><div class="stat-pill-val">{shooting}</div></div>
                <div class="stat-pill"><div class="stat-pill-label">PAS</div><div class="stat-pill-val">{passing}</div></div>
                <div class="stat-pill"><div class="stat-pill-label">DRI</div><div class="stat-pill-val">{dribbling}</div></div>
                <div class="stat-pill"><div class="stat-pill-label">DEF</div><div class="stat-pill-val">{defending}</div></div>
                <div class="stat-pill"><div class="stat-pill-label">PHY</div><div class="stat-pill-val">{physic}</div></div>
            </div>
        </div>
        """
        st.html(textwrap.dedent(card_html))

        # Tactical Radar Chart
        categories = ['Pace', 'Shooting', 'Passing', 'Dribbling', 'Defending', 'Physicality']
        values = [pace, shooting, passing, dribbling, defending, physic]
        values += values[:1]  # close polygon
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(4.0, 3.2), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor('#0F172A')
        ax.set_facecolor('#0F172A')

        ax.plot(angles, values, color='#10B981', linewidth=2, linestyle='solid')
        ax.fill(angles, values, color='#10B981', alpha=0.35)

        ax.set_ylim(0, 100)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, color='#94A3B8', size=9, weight='bold')
        ax.set_yticklabels([])
        ax.spines['polar'].set_color('#334155')
        ax.grid(color='#334155', linestyle='--', alpha=0.7)

        plt.tight_layout()
        st.pyplot(fig)

    # -------------------------------------------------------------------------
    # AI SCOUT TWINS
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🔍 AI Scout Match Engine (Closest Player Twins in Database)")
    st.markdown("Nearest-neighbor search using cosine similarity across 8 physical and tactical skills:")

    scout_vector = pd.DataFrame([[pace, shooting, passing, dribbling, defending, physic, overall, potential]], columns=scout_cols)
    scout_vector_scaled = scout_scaler.transform(scout_vector)
    dists, indices = nn_model.kneighbors(scout_vector_scaled, n_neighbors=4)

    twin_cols = st.columns(3)
    for i, idx in enumerate(indices[0][1:4]):
        matched_player = player_db.iloc[idx]
        sim_pct = max(0.0, (1.0 - dists[0][i+1])) * 100.0
        
        with twin_cols[i]:
            twin_html = f"""
            <div class="twin-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="background: rgba(56, 189, 248, 0.15); color: #38BDF8; font-weight: 800; font-size: 0.8rem; padding: 2px 8px; border-radius: 6px;">
                        #{i+1} MATCH: {sim_pct:.1f}%
                    </span>
                    <span style="color: #94A3B8; font-size: 0.85rem;">Age {matched_player['age']}</span>
                </div>
                <div style="font-size: 1.2rem; font-weight: 800; color: #F8FAFC; margin: 8px 0 2px 0;">
                    {matched_player['short_name']}
                </div>
                <div style="color: #94A3B8; font-size: 0.85rem;">
                    {matched_player['position_group']} · OVR: <strong style="color: #FBBF24;">{matched_player['overall']}</strong> · POT: <strong>{matched_player['potential']}</strong>
                </div>
                <div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid #1E293B; display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.75rem; color: #94A3B8;">MARKET VALUE</span>
                    <span style="font-size: 1.1rem; font-weight: 800; color: #34D399;">€{matched_player['value_eur'] / 1e6:.1f}M</span>
                </div>
            </div>
            """
            st.html(textwrap.dedent(twin_html))

# =============================================================================
# TAB 2: MODEL PERFORMANCE & BENCHMARKS
# =============================================================================
with tab_analytics:
    st.markdown("### 📊 Model Evaluation & Engineering Benchmarks (FIFA 2022)")
    st.markdown("Evaluating our tuned pipeline against alternative regressors on **2,875 unseen test players**.")

    # High Level KPI Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.html(textwrap.dedent("""
        <div class="kpi-card">
            <div class="kpi-num">0.9989</div>
            <div class="kpi-lbl">Log R² Score</div>
        </div>
        """))
    with kpi2:
        st.html(textwrap.dedent("""
        <div class="kpi-card">
            <div class="kpi-num" style="color: #34D399;">99.69%</div>
            <div class="kpi-lbl">Accuracy within ±20%</div>
        </div>
        """))
    with kpi3:
        st.html(textwrap.dedent("""
        <div class="kpi-card">
            <div class="kpi-num" style="color: #FBBF24;">€13,926</div>
            <div class="kpi-lbl">Median Absolute Error</div>
        </div>
        """))
    with kpi4:
        st.html(textwrap.dedent("""
        <div class="kpi-card">
            <div class="kpi-num">0.44s</div>
            <div class="kpi-lbl">Pipeline Training Time</div>
        </div>
        """))

    st.markdown("---")

    # Benchmark Table
    bench_data = {
        'Algorithm': ['Tuned LightGBM (Champion)', 'Tuned XGBoost', 'CatBoost', 'Random Forest'],
        'Log R²': [0.9989, 0.9989, 0.9986, 0.9980],
        'Orig R²': [0.9687, 0.9689, 0.9831, 0.9526],
        'MAE (€)': ['€111,772', '€106,147', '€91,149', '€149,553'],
        'MedAE (€)': ['€13,926', '€15,237', '€17,256', '€12,106'],
        'Valuation Acc (±20%)': ['99.69%', '99.51%', '99.65%', '99.10%'],
        'Train Time (s)': ['0.44s', '0.57s', '1.08s', '1.32s']
    }
    st.dataframe(pd.DataFrame(bench_data), hide_index=True, width='stretch')

    # Visual Figures
    st.markdown("#### 📈 Key Visualizations")
    fig_col1, fig_col2 = st.columns(2)
    
    act_pred_path = "evaluation_results/figures/actual_vs_predicted.png"
    feat_imp_path = "evaluation_results/figures/top20_feature_importance.png"

    with fig_col1:
        if os.path.exists(act_pred_path):
            st.image(act_pred_path, caption="Actual vs Predicted Market Valuation (Test Set)", width='stretch')
    with fig_col2:
        if os.path.exists(feat_imp_path):
            st.image(feat_imp_path, caption="Top Feature Importance (Feature Gain %)", width='stretch')
