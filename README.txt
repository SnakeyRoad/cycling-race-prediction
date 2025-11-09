================================================================================
CYCLING RACE PREDICTION - ML PROJECT REPOSITORY
================================================================================

Project: ML Prediction of Top 30 Finishers in Cycling Races
Team Members: Chris, David, Youri, Iris, Christian
Date: October-November 2025
Course: Machine Learning (MA01)

================================================================================
OVERVIEW
================================================================================

This repository contains a complete machine learning pipeline for predicting
whether professional cyclists will finish in the top 30 positions of races,
which determines UCI point eligibility. The project follows CRISP-DM methodology
and includes data cleaning, preprocessing, multiple modeling approaches, and
comprehensive evaluation.

Target Variable: 
  - 1 = Top 30 finish (earns UCI points)
  - 0 = Outside top 30

Dataset: Professional cycling race results from 2017-2021 (~125,000 records)

================================================================================
REPOSITORY STRUCTURE
================================================================================

cycling-race-prediction/
│
├── README.txt                          # This file
├── cycling_big.db                      # SQLite database with raw race data
│
├── CLEANED_DATA/                       # Preprocessed datasets ready for modeling
│   ├── cleaned_data_2017.csv          # 2017 data (~26,700 rows, 18 features)
│   ├── cleaned_data_2018.csv          # 2018 data (~24,400 rows, 18 features)
│   ├── cleaned_data_2019.csv          # 2019 data (~24,800 rows, 18 features)
│   ├── cleaned_data_2020.csv          # 2020 data (~16,000 rows, 18 features)
│   └── cleaned_data_2021.csv          # 2021 data (~28,200 rows, 18 features)
│
├── DATA_CLEANING/                      # Data preprocessing pipeline
│   └── cycling_preprocessing.ipynb    # Complete preprocessing notebook
│
└── MODELS/                             # Machine learning models by team members
    ├── catboost_cycling_model.ipynb   # CatBoost gradient boosting (Chris)
    ├── lightgbm_finetuning.ipynb      # LightGBM hyperparameter tuning
    ├── ANN_MODELLING.ipynb            # Artificial Neural Network approach
    └── model_comparison.ipynb         # Comparative analysis of all models

Total: ~125,000 training examples across 5 years

================================================================================
DATA FILES
================================================================================

1. CYCLING_BIG.DB (SQLite Database)
-----------------------------------
Contains raw race results from 2012-2021 including:
- race_results table: Stage-by-stage results
- rider_infos table: Rider physical stats and specialties
- Missing: 2013 data (never collected)

2. CLEANED_DATA FILES (CSV Format)
----------------------------------
Each year's CSV contains 18 columns:
- 1 target variable (binary classification)
- 17 features (numeric, preprocessed)
- No missing values
- Ready for machine learning
- Memory-efficient float16 format

See "DATASET STRUCTURE" section below for detailed column descriptions.

================================================================================
NOTEBOOKS
================================================================================

DATA_CLEANING/
--------------

cycling_preprocessing.ipynb
- Complete preprocessing pipeline from raw data to clean CSVs
- Handles missing 2013 data (excludes 2016 accordingly)
- Creates historical performance features (3-year lookback)
- Imputes missing values, one-hot encodes categoricals
- Produces 5 cleaned CSV files (2017-2021)
- Includes validation checks and documentation

MODELS/
-------

catboost_cycling_model.ipynb 
- CatBoost gradient boosting implementation
- Temporal train/validation/test split (2017-2019/2020/2021)
- Feature importance analysis
- Performance: 80.94% ROC-AUC, 81.98% accuracy on test set
- Handles class imbalance with balanced weights
- Key finding: GC position and sumres_1 are most predictive features

lightgbm_finetuning.ipynb
- LightGBM implementation with hyperparameter optimization
- Grid search or Bayesian optimization approach
- Comparison with CatBoost performance
- Feature engineering experiments

ANN_MODELLING.ipynb
- Artificial Neural Network approach using deep learning
- Architecture exploration (layers, neurons, activation functions)
- Dropout and regularization for overfitting prevention
- Comparison with tree-based methods

model_comparison.ipynb
- Comprehensive comparison across all modeling approaches
- Performance metrics: accuracy, ROC-AUC, precision, recall, F1
- Feature importance across different models
- Ensemble method exploration
- Final model selection and justification

================================================================================
DATASET STRUCTURE
================================================================================

Each cleaned CSV file contains 18 columns:

TARGET VARIABLE:
- target (int): 1 if rider finished in top 30, else 0

FEATURES (17 total):

Race Context:
- GC (float): General Classification position (most important feature)
- Age (float): Rider's age
- Length (float): Race stage length in km
- Stage_Type_RR (float): 1 if road race, 0 otherwise (one-hot encoded)

Physical Attributes:
- height (float): Rider height in meters
- weight (float): Rider weight in kg

Specialty Scores (career performance by race type):
- One day races (float): Points from one-day races
- GC_specialty (float): Points from general classification races
- Time trial (float): Points from time trials
- Sprint (float): Points from sprint stages
- Climber (float): Points from climbing stages

Rankings:
- PCS Ranking (float): ProCyclingStats ranking
- UCI World Ranking (float): UCI official ranking
- Specials | All Time Ranking (float): All-time ranking

Historical Performance (3-year lookback):
- sumres_1 (float): Total UCI points earned previous year (highly predictive)
- sumres_2 (float): Total UCI points earned 2 years ago
- sumres_3 (float): Total UCI points earned 3 years ago

Class Distribution: ~21% top 30 finishers, ~79% outside top 30

================================================================================
WHY 2016 IS EXCLUDED (CRITICAL INFORMATION)
================================================================================

Historical features require 3 years of previous race data:
- sumres_1: Points earned in previous year (Y-1)
- sumres_2: Points earned 2 years ago (Y-2)
- sumres_3: Points earned 3 years ago (Y-3)

For 2016 predictions:
- sumres_1 needs 2015 data ✓ (available)
- sumres_2 needs 2014 data ✓ (available)
- sumres_3 needs 2013 data ✗ (MISSING - never collected)

Investigation confirmed that 2013 data does not exist in cycling_big.db.
Without 2013 data, every rider in 2016 would have incomplete historical features.

Decision: Exclude 2016 to maintain data quality and feature consistency.
Impact: Lose ~25,000 rows but gain complete, reliable features across all years.

================================================================================
DATA QUALITY CHECKS PERFORMED
================================================================================

✓ Removed riders with team = "noteam"
✓ Filtered out non-numeric ranks (DNF, OTL, DNS, etc.)
✓ Converted Length from "118.5 km" string to numeric 118.5
✓ Parsed JSON-like strings in rider_infos (pps and rdr columns)
✓ Merged rider information with race results
✓ Calculated 3-year historical performance features
✓ Imputed missing values with column means
✓ Dropped rows with critical missing data
✓ Created binary target variable (top 30 threshold)
✓ One-hot encoded categorical variables (Stage_Type)
✓ Validated no missing values in final datasets
✓ Converted to float16 for memory efficiency

================================================================================
TECHNICAL SPECIFICATIONS
================================================================================

Development Environment:
- OS: Linux Mint
- IDE: VSCodium
- Python: 3.12.8 (virtual environment)
- Git: SSH authentication for GitHub

Required Packages:
- pandas: Data manipulation
- numpy: Numerical operations
- scikit-learn: ML utilities and metrics
- catboost: Gradient boosting (primary model)
- lightgbm: Alternative gradient boosting
- tensorflow/keras: Neural network implementation
- matplotlib/seaborn: Visualization
- jupyter: Notebook environment

Hardware:
- HP EliteBook 840 G8 Notebook PC

Processing Time:
- Full preprocessing pipeline: ~2 minutes
- CatBoost training: ~30 seconds per fold
- Memory usage: ~500MB peak

File Sizes:
- cycling_big.db: ~180MB (SQLite database)
- Each cleaned CSV: ~1.5MB (float16 compression)
- Total repository: ~190MB

================================================================================
TROUBLESHOOTING
================================================================================

1. DATA LOADING ISSUES
   - Ensure you're in the repository root directory
   - Use relative paths: 'CLEANED_DATA/cleaned_data_2019.csv'
   - Check file exists with: ls CLEANED_DATA/

2. MISSING VALUES ERROR
   - All cleaned CSVs have no missing values
   - If you see NaN, check your data loading code
   - Verify correct CSV file name and path

3. MEMORY ISSUES
   - Data uses float16 to minimize memory
   - Load one year at a time if needed
   - Close other applications during model training

4. CLASS IMBALANCE
   - Target naturally imbalanced (21% vs 79%)
   - Use class_weight='balanced' in sklearn models
   - Use scale_pos_weight in CatBoost/LightGBM
   - Consider stratified sampling for validation

================================================================================
TEAM RESPONSIBILITIES
================================================================================


- CatBoost gradient boosting implementation
- Feature importance analysis
- Temporal validation strategy
- Performance: 80.94% ROC-AUC, 81.98% accuracy


- Project requirements and analysis guidance
- Correlation analysis and outlier detection coordination
- Quality assurance and methodology validation


- Alternative modeling approaches (Logistic Regression, Time Series)
- Model comparison contributions
- Name matching tasks


- Data preprocessing pipeline
- Historical feature engineering
- Data quality validation and documentation

================================================================================
NEXT STEPS & FUTURE WORK
================================================================================

1. MODEL OPTIMIZATION
   - Hyperparameter tuning (careful of overfitting)
   - Feature selection based on importance analysis
   - Ensemble methods combining multiple models

2. ANALYSIS DEEPENING
   - Correlation matrix and multicollinearity analysis
   - Outlier detection and treatment
   - Feature interaction exploration

3. REPORT PREPARATION
   - Results section completion (3 subsections, 1.5 pages each)
   - Model comparison tables and visualizations
   - Academic writing with strict 10-page limit
   - Professional formatting (narrow margins, compact style)

4. POTENTIAL IMPROVEMENTS
   - Include team-level features (team strength, budget)
   - Weather conditions if available
   - Race-specific difficulty ratings
   - Rider injury history

================================================================================
PROJECT DELIVERABLES
================================================================================

Academic Report Structure:
1. Introduction
2. Methodology (CRISP-DM)
3. Results
   3.1 Data Preparation (~1.5 pages)
   3.2 Modelling (~1.5 pages)
   3.3 Evaluation (~1.5 pages)
4. Discussion
5. Conclusion

Constraints:
- Total: 10 pages maximum
- Narrow margins, compact formatting
- Professional academic style
- Comprehensive but concise

Repository Contents:
✓ Clean, well-documented code
✓ Preprocessed datasets ready for modeling
✓ Multiple modeling approaches
✓ Comprehensive README documentation
✓ Reproducible results

================================================================================
REFERENCES & DATA SOURCES
================================================================================

Data Source: ProCyclingStats (via web scraping)
Database: SQLite (cycling_big.db)
Methodology: CRISP-DM (Cross-Industry Standard Process for Data Mining)
Version Control: GitHub (https://github.com/SnakeyRoad/cycling-race-prediction)

Course: Machine Learning MA01
Institution: ADSAI
Academic Year: 2025-2026

================================================================================
CONTACT INFORMATION
================================================================================

For questions about:
- Preprocessing: Christian (cpm)
- CatBoost modeling: Chris
- Project coordination: David
- Other models: David, Youri, Iris

Repository: https://github.com/SnakeyRoad/cycling-race-prediction
Project Directory: /home/cpm/Desktop/ML_PROJECT/

================================================================================
VERSION HISTORY
================================================================================

v2.0 - November 2025
- Complete repository restructure with organized folders
- Added all team modeling notebooks (CatBoost, LightGBM, ANN, comparison)
- Included cycling_big.db SQLite database
- Updated documentation with model results and findings
- Prepared for final academic report submission

v1.0 - October 2025
- Initial preprocessing pipeline created
- 5 years of cleaned data (2017-2021)
- Basic documentation and validation
- Ready for team modeling phase

================================================================================
LICENSE & ACADEMIC INTEGRITY
================================================================================

This project is submitted as part of coursework for Machine Learning MA01.
All code and analysis are original work by the team members listed above.
Data sourced from publicly available ProCyclingStats website.

For academic use only. Not for commercial distribution.

================================================================================
ACKNOWLEDGMENTS
================================================================================

- Team Members: Chris, David, Youri, Iris, Christian
- Data Source: ProCyclingStats
- Course Instructors: [MA01 Teaching Team]
- Development Tools: Python, scikit-learn, CatBoost, Jupyter

================================================================================
END OF DOCUMENTATION
================================================================================
