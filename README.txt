================================================================================
BOOSTME: CYCLING RACE PREDICTION - ML PROJECT
================================================================================

Project: Machine Learning Prediction of UCI Point-Earning Race Finishes
Team: Youri van der Meulen, David Fernandez Santamonica, 
      Christian Mol, Iris Walraven Bartolome
Date: November 2025
Course: Machine Learning (MA01)
Institution: Zuyd University
GitHub: https://github.com/SnakeyRoad/cycling-race-prediction

================================================================================
PROJECT OVERVIEW
================================================================================

CAICLE, an investment firm building a professional cycling super team, requires
AI-powered models to predict race performance for talent recruitment. This project
develops multiple machine learning models to identify riders likely to earn UCI
points through strong race finishes, enabling data-driven recruitment decisions.

Business Goal: Predict which cyclists will finish in top 30 positions (UCI points)
Research Question: How can ML models predict cycling performance from historical data?
Approach: CRISP-DM methodology with emphasis on data preparation, modeling, and evaluation

Dataset: 225,918 race results from 2012-2021 → 120,261 usable records (2017-2021)
Target: Binary classification (UCI point-earning finish vs non-earning)
Class Distribution: 20.4% point-earning, 79.6% non-earning

================================================================================
REPOSITORY STRUCTURE
================================================================================

cycling-race-prediction/
│
├── README.txt                              # This file
├── cycling_big.db                          # SQLite database (raw data)
│
├── CLEANED_DATA/                           # Preprocessed datasets
│   ├── cleaned_data_2017.csv              # 26,698 rows, 18 features
│   ├── cleaned_data_2018.csv              # 24,417 rows, 18 features
│   ├── cleaned_data_2019.csv              # 24,849 rows, 18 features
│   ├── cleaned_data_2020.csv              # 16,056 rows, 18 features (pandemic)
│   └── cleaned_data_2021.csv              # 28,241 rows, 18 features
│
├── DATA_CLEANING/
│   └── cycling_preprocessing.ipynb        # Complete preprocessing pipeline
│
└── MODELS/
    ├── catboost_cycling_model.ipynb       # CatBoost (best: 80.94% ROC-AUC)
    ├── lightgbm_finetuning.ipynb          # LightGBM (82.3% balanced accuracy)
    ├── ANN_MODELLING.ipynb                # Neural networks (10 architectures)
    ├── model_comparison.ipynb             # Multi-model comparison
    ├── random_forest_advanced.pkl         # Trained Random Forest
    ├── xgboost.pkl                        # Trained XGBoost
    └── complete_prediction_pipeline.py    # Production pipeline

================================================================================
KEY FINDINGS
================================================================================

BEST MODELS:
1. CatBoost: 80.94% test ROC-AUC, 73.7% recall, 62.3% F1 → Comprehensive scouting
2. LightGBM: 82.3% balanced accuracy, 77.1% recall → Fast retraining
3. XGBoost: 87.2% ROC-AUC → Production stability

OTHER APPROACHES:
- Random Forest: 72% balanced accuracy, 50.9% recall → High interpretability
- Logistic Regression: 82.1% ROC-AUC → Transparent, explainable
- Neural Networks: Best F1 59.6% → Promising for future rich data

FEATURE IMPORTANCE (All Models):
1. GC (General Classification position) - Dominant predictor
2. sumres_1 (Previous year UCI points) - Recent performance
3. UCI World Ranking - Professional standing
4. PCS Ranking - Alternative ranking system

Key Insight: Recent form and tactical positioning matter more than physical attributes

================================================================================
DATA PREPARATION HIGHLIGHTS
================================================================================

TEMPORAL DATA SPLITTING (Critical for preventing leakage):
- Training: 2017-2019 (75,964 records)
- Validation: 2020 (16,056 records)
- Test: 2021 (28,241 records)

Initial random splitting inflated ROC-AUC to ~0.82 (data leakage)
Proper temporal splitting reduced to realistic ~0.81 (true generalization)

WHY 2016 IS EXCLUDED:
Historical features require 3-year lookback (sumres_1, sumres_2, sumres_3)
For 2016: Need 2015✓, 2014✓, 2013✗ (MISSING from database)
Decision: Exclude 2016 to maintain complete, consistent features

PREPROCESSING PIPELINE:
- 571,772 initial missing values → KNN imputation (k=5) → 0 missing
- JSON parsing for specialty scores (85% recovery rate)
- Outlier detection (Z>3): 99 flagged, 12 corrected
- Feature reduction: 43 raw features → 18 final features
- Dropped sumres_2 and sumres_3 (no added predictive value beyond sumres_1)

================================================================================
DATASET STRUCTURE (18 Features)
================================================================================

TARGET:
- target: 1 if UCI point-earning finish, 0 otherwise

RACE CONTEXT (4 features):
- GC: General Classification position (most important)
- Age: Rider age
- Length: Stage length (km)
- Stage_Type_RR: Binary road race indicator

PHYSICAL (2 features):
- height: Rider height (m)
- weight: Rider weight (kg)

SPECIALTY SCORES (5 features):
- One day races, GC_specialty, Time trial, Sprint, Climber

RANKINGS (3 features):
- PCS Ranking, UCI World Ranking, All Time Ranking

HISTORICAL PERFORMANCE (1 feature after ablation):
- sumres_1: Total UCI points from previous year
(Note: sumres_2 and sumres_3 removed after showing no predictive improvement)

================================================================================
MODEL RECOMMENDATIONS
================================================================================

TIERED IMPLEMENTATION STRATEGY:

Use CatBoost/LightGBM when:
- Comprehensive talent scouting (maximize candidate identification)
- High recall is critical (can't miss potential talent)
- Fast retraining needed

Use Random Forest when:
- Explaining predictions to non-technical stakeholders
- Interpretability > maximum performance
- Stakeholder communication is priority

Use Logistic Regression when:
- Legally defensible decisions required
- Regulatory compliance needs explainability
- Resource-constrained scenarios

Use XGBoost when:
- Production-grade stability needed
- System integration is priority
- Extensive community support valuable

================================================================================
LIMITATIONS & FUTURE WORK
================================================================================

CURRENT LIMITATIONS:
- Heavy dependence on in-race features (GC) limits pre-race prediction
- Class imbalance produces false positives requiring human screening
- 2020 validation may be unreliable (pandemic disruptions)
- Model excludes team dynamics, tactics, course profiles

CRITICAL IMPROVEMENTS (Aligned with CAICLE's Vision):
1. Integrate physiological metrics:
   - FTP (Functional Threshold Power)
   - Power output (watts/kg)
   - Heart rate, cadence, elevation gain
   → Better predictors than placement history alone

2. Expand to amateur talent pools:
   - Outstanding riders often emerge from lower leagues
   - BoostMe platform: Free AI training for performance data
   → Unique data advantage over competitors

3. Technical enhancements:
   - Ensemble methods (combine multiple models)
   - External data (team strength, weather, head-to-head)
   - Web application for scout accessibility

4. Human-AI collaboration:
   - Augment (not replace) human scouts
   - Assess intangibles: leadership, team chemistry
   - Balance quantitative + qualitative insights

================================================================================
EVALUATION METHODOLOGY
================================================================================

PRIMARY METRICS (Imbalanced Classification):
- ROC-AUC: Discriminative ability (ranking point-earning vs non-earning)
- Recall (Class 1): Proportion of actual point-earners identified
- F1-Score (Class 1): Balance of precision and recall
- Balanced Accuracy: Account for class distribution

WHY NOT SIMPLE ACCURACY:
With 79.6% non-earning riders, a model predicting "no" for everyone achieves
79.6% accuracy but identifies 0% of talent → Useless for business

BALANCED ACCURACY LIMITATION:
Equal weighting (50% each class) misaligns with business needs where missing
talent (false negatives) costs more than wasted scouting (false positives)
→ Prioritize recall and F1 over balanced accuracy for deployment

================================================================================
TECHNICAL SPECIFICATIONS
================================================================================

Development:
- Python 3.12.8 in virtual environment
- Linux Mint, VSCodium IDE
- Git SSH authentication

Key Libraries:
- pandas, numpy: Data manipulation
- scikit-learn: ML utilities, metrics
- catboost, lightgbm, xgboost: Gradient boosting
- tensorflow/keras: Neural networks
- matplotlib, seaborn: Visualization

Processing Performance:
- Full preprocessing: ~2 minutes
- CatBoost training: ~13 seconds
- Memory: ~380MB peak

================================================================================
BUSINESS INTERPRETATION
================================================================================

From CAICLE's perspective:
- 74-81% recall captures most point-earning riders
- High-recall models (CatBoost, LightGBM) for broad candidate screening
- High-precision models (Random Forest) for confident final selections
- Recent performance (sumres_1) best indicates future success
- Threshold tuning balances false positives vs missed talent

Model Portfolio Enables:
- Comprehensive talent identification (high recall)
- Confident recruitment decisions (high precision)
- Explainable predictions (interpretable models)
- Flexible deployment (speed vs performance trade-offs)

================================================================================
ACADEMIC DELIVERABLE
================================================================================

Extended Abstract Structure:
1. Introduction: Business context, research question
2. Method: CRISP-DM, team approach, temporal validation
3. Results:
   3.1 Data Preparation: Cleaning, features, temporal splitting
   3.2 Modeling: Multiple approaches (gradient boosting, neural networks)
   3.3 Evaluation: Metrics, optimization, business interpretation
4. Conclusion: Model diversity, recommendations, future work

Constraints:
- Emphasis on interpretability and business context
- Multiple models for different scenarios (not single "best")
- Model quality = discriminative ability + precision + recall + efficiency
- Success defined by business needs, not just statistical metrics

================================================================================
REFERENCES
================================================================================

GitHub Repository: https://github.com/SnakeyRoad/cycling-race-prediction

Academic Sources:
[1] J. Schmidhuber, "Deep Learning in Neural Networks," arXiv:1404.7828, 2014
[2] Z. Zhu et al., "Robustness in deep learning," arXiv:2209.07263, 2022
[3] C. Hettinger, "Hyperparameters for Dense Networks," BYU M.S. thesis, 2019
[4] N. Phelps et al., "Challenges learning from imbalanced data," arXiv:2412.16209, 2024

Data Source: ProCyclingStats (via SQLite database)
Methodology: CRISP-DM framework

================================================================================
TEAM CONTRIBUTIONS
================================================================================

Startup Approach: All members contributed across phases rather than fixed roles

Supervisors: M. Vaessen, J. Baljan, B. Kroon, K. Manjari

================================================================================
VERSION HISTORY
================================================================================

v2.0 - November 2025
- Complete multi-model implementation (CatBoost, LightGBM, XGBoost, RF, ANN)
- Extended abstract with comprehensive results
- Business-focused recommendations and tiered deployment strategy
- Future work aligned with BoostMe platform vision

v1.0 - October 2025
- Initial preprocessing pipeline
- 5 years cleaned data (2017-2021)
- Temporal validation strategy
- Ready for modeling phase

================================================================================
CONTACT & SUPPORT
================================================================================

- Individual team members can be reached via their university e-mail
- Any questions can be asked during presentation or panel discussion

Repository: https://github.com/SnakeyRoad/cycling-race-prediction
Course: Machine Learning MA01, Zuyd University

================================================================================
LICENSE
================================================================================

Academic project for Machine Learning MA01 coursework
Original work by team members (Youri, David, Christian, Iris)
Data from publicly available ProCyclingStats
For educational use only

================================================================================
END OF DOCUMENTATION
================================================================================
