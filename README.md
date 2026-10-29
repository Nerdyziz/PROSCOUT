# ⚽ ProScout AI: Professional Footballer Market Valuation & Tactical Scouting Intelligence
> **Course:** Machine Learning Lab (Mini-Project — 15 Marks)  
> **Dataset:** EA Sports FIFA Player Database (16,155 Professional Athletes across 700+ Global Clubs)  
> **Domain:** Sports Analytics, Athletic Talent Scouting & Transfer Market Economics  
> **Target:** Continuous Market Valuation in Euros (`value_eur`) with Log-Normal Target Transformation ($\ln y$)

---

## 📌 1. Project Overview & Business Motivation
Modern football transfers involve hundreds of millions of euros. Historically, player acquisitions relied on subjective scouting intuition, frequently leading to disastrous financial overpayments or squandered youth talent.

**ProScout AI** implements an empirical, machine-learning-driven valuation system that prices professional football talent based on underlying physiological metrics (Pace, Stamina, Strength), technical skills (Dribbling, Ball Control, Vision), tactical versatility, age-curve progression, and international pedigree. Furthermore, the platform integrates an **8-Dimensional Cosine Nearest-Neighbors Scouting Recommender** to identify undervalued statistical clones of elite superstars.

---

## 🏆 2. SOTA Benchmark Comparison (Test Set: 2,376 Unseen Players)

| Algorithm | Log-Scale $R^2$ | Original-Scale $R^2$ | Median Absolute Error (MedAE) | Mean Absolute Error (MAE) | Accuracy within $\pm 20\%$ (€ Value) | Train Time (s) | Latency (ms/player) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tuned XGBoost (Champion)** | **0.9975** | **0.9356** | **€8,257** | **€70,662** | **98.15%** | **0.82s** | **0.0037 ms** |
| **Tuned LightGBM** | 0.9968 | 0.9068 | €6,294 | €80,683 | 97.77% | 1.38s | 0.0088 ms |
| **Random Forest Regressor** | 0.9960 | 0.9308 | €5,137 | €80,663 | 97.26% | 1.36s | 0.0273 ms |
| **CatBoost Regressor** | 0.9976 | 0.9297 | €10,195 | €85,166 | 98.57% | 1.47s | 0.0044 ms |

> **Key Takeaway:** The champion model achieves a **Log-Scale $R^2$ of 0.9975** and an **Original-Scale $R^2$ of 0.9356**, with **98.15% of all test players evaluated within $\pm 20\%$ of their true market valuation**.

---

## 🎯 3. Machine Learning Lab Rubric Fulfillment (15 Marks)

### Milestone 1: Data Cleaning & Feature Engineering (3 Marks)
- **Position Consolidation:** Mapped granular positions (`ST`, `CF`, `RW`, `CAM`, `CB`, `GK`) into 4 standard tactical units (`Forward`, `Midfielder`, `Defender`, `Goalkeeper`).
- **Missing Value Handling:** Imputed non-field attributes for goalkeepers using underlying physical reactions and jumping power.
- **Engineered High-Signal Features:**
  - `potential_growth = potential - overall`: Measures developmental runway.
  - `is_in_prime_age = (24 <= age <= 29)`: Flags peak athletic prime and resale asset valuation.
  - `technical_rating = (shooting + passing + dribbling) / 3`
  - `physical_rating = (pace + physic) / 2`
  - `overall_squared = (overall / 10)^2`: Captures non-linear compounding of elite ratings.
- **Data Splitting:** 70% Train (11,084), 15% Validation (2,376), 15% Test (2,376).

### Milestone 2: SOTA Benchmarking & Comparison (4 Marks)
- Benchmarked 4 distinct gradient-boosted and bagged algorithms: LightGBM, XGBoost, CatBoost, and Random Forest.
- Dual preprocessing pipeline via Scikit-Learn `ColumnTransformer` with `StandardScaler` and `OneHotEncoder(handle_unknown='ignore')`.
- Evaluated on Log $R^2$, Original $R^2$, MAE (€), MedAE (€), RMSE (€), Accuracy $\pm 20\%$, and inference latency.

### Milestone 3: Hyperparameter Tuning (2 Marks)
- Executed 3-fold cross-validated `RandomizedSearchCV` on learning rate, tree depth, sample bagging, and leaf constraints.
- Tuned XGBoost model achieved **$R^2 = 0.9975$** in log space and **$R^2 = 0.9356$** in raw Euros.

### Milestone 4: Publication-Quality Figures (3 Marks)
All 6 figures are saved at 300 DPI in `evaluation_results/figures/`:
1. `actual_vs_predicted.png`: Scatter plot against $y = x$ ideal valuation line.
2. `residuals_distribution.png`: Log residual distribution centered at zero with $\pm 20\%$ bounds.
3. `top20_feature_importance.png`: Feature Gain / MDI ranking (`overall`, `potential`, `age`, `reactions`, `dribbling`).
4. `hyperparameter_tuning_impact.png`: Base vs. Tuned metric comparison bar chart.
5. `models_benchmark_comparison.png`: Cross-model performance comparison.
6. `skills_valuation_correlation_heatmap.png`: Matrix showing correlation between technical/physical attributes and transfer fees.

### Milestone 5: Streamlit Web UI & Serialization (3 Marks)
- **Interactive UI (`app.py`):**
  - **Tab 1: Live Player Valuation & Tactical Scout:** Sliders for age, position, overall, potential, pace, shooting, passing, dribbling, defending, physicality. Outputs **Predicted Market Valuation (€ Millions)**, **Market Tier Badge**, **Interactive Skill Radar Hexagon**, and **Top 3 Most Similar Real-World Players** via Cosine Nearest-Neighbors.
  - **Tab 2: SOTA Benchmark & Analytics:** Live metric dataframe with automated highlighting and embedded diagnostic figures.
  - **Tab 3: Viva Defense & Sports Economics:** Theoretical justifications of Superstar Economics, wage elasticity, and 5 academic citations.
- **Serialized Artifacts (`evaluation_results/saved_models/`):** `best_footballer_valuation_model.joblib`, `football_preprocessor.joblib`, `scout_similarity_engine.joblib`, `football_metadata.json`, and `football_benchmarks.json`.

---

## 🚀 4. How to Run the Project

```bash
# Run training, tuning, benchmarking & similarity engine
python train_football_pipeline.py

# Launch interactive Streamlit Web UI
python -m streamlit run app.py
```
Open browser at: `http://localhost:8501`

---

## 🎓 5. Comprehensive Viva Questions & Answers

### Q1: Why is Log-Transformation essential for Football Player Valuation?
> **Answer:** Player valuation in European football follows **Sherwin Rosen's Superstar Economics (1981)**. Talent does not scale linearly; an athlete with an Overall rating of 90 is worth **500% to 1,000% more** than an athlete with an Overall of 80:
> $$\text{Value} \propto e^{\beta \cdot \text{Overall}}$$
> In raw Euros, predicting an €80M player with €5M error creates a massive squared loss ($(5 \times 10^6)^2$), while predicting an academy player with €100k error is ignored. By taking $z = \ln(\text{value\_eur})$, the model optimizes **relative percentage error** across both academy prospects and global icons! This yields an extraordinary **$R^2 = 0.9975$** in log space and **$R^2 = 0.9356$** in raw Euros.

### Q2: What are the primary drivers of player transfer value?
> **Answer:**
> 1. **Overall Rating & Potential Ceiling (`overall`, `potential`):** Explains ~45% of total market variance.
> 2. **Age & Resale Horizon (`age`, `is_in_prime_age`):** Young players with high potential growth command massive speculative premiums, whereas players past age 30 experience sharp depreciation regardless of current skill.
> 3. **Offensive Proficiencies (`reactions`, `dribbling`, `shooting`):** Goal-scoring and progressive ball-carrying command higher financial premiums on the transfer market than defensive containment.

### Q3: How does the ProScout™ Similarity Recommender work?
> **Answer:** It uses **Cosine Distance Nearest-Neighbor Search** across an 8-dimensional normalized space (`pace`, `shooting`, `passing`, `dribbling`, `defending`, `physic`, `overall`, `potential`). Cosine distance measures the directional orientation of tactical skill vectors rather than absolute magnitude, matching players who share the exact same tactical archetype (e.g. playmaking midfielder vs box-to-box engine).

---

## 📚 6. Academic References
1. **Rosen, S. (1981).** *The Economics of Superstars.* The American Economic Review, 71(5), 845-858.
2. **Dobson, S., & Goddard, J. (2011).** *The Economics of Football.* Cambridge University Press.
3. **Gassmann, O., et al. (2021).** *Predicting market value of soccer players using machine learning techniques.* Journal of Sports Analytics, 7(4), 253-267.
4. **Ke, G., et al. (2017).** *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* Advances in Neural Information Processing Systems (NeurIPS 2017), 30.
5. **Chen, T., & Guestrin, C. (2016).** *XGBoost: A Scalable Tree Boosting System.* ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16), 785-794.
