# COMPLETE CYCLING RACE PREDICTION PIPELINE
# Includes: Data Loading, Preprocessing, Multiple Models, Comparison, and Business Insights

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (roc_auc_score, precision_score, recall_score, f1_score, 
                           classification_report, roc_curve, confusion_matrix, 
                           precision_recall_curve, average_precision_score)
from sklearn.preprocessing import StandardScaler
import joblib
import time
import os
import warnings
warnings.filterwarnings('ignore')

print(" COMPLETE CYCLING RACE PREDICTION ANALYSIS")
print("=" * 70)

# Create all necessary directories
os.makedirs('results/comprehensive', exist_ok=True)
os.makedirs('models', exist_ok=True)

# 1: LOAD AND PREPARE DATA


print("\n STEP 1: LOADING AND PREPARING DATA")
print("=" * 50)

# Load all cleaned years
years = [2017, 2018, 2019, 2020, 2021]
dataframes = []

for year in years:
    try:
        df_year = pd.read_csv(f'cleaned_data_{year}.csv')
        dataframes.append(df_year)
        print(f"✅ Loaded {year}: {len(df_year)} rows")
    except FileNotFoundError:
        print(f"❌ File for {year} not found")

# Combine all data
df = pd.concat(dataframes, ignore_index=True)

# Prepare features and target
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📈 DATASET SUMMARY:")
print(f"   Total samples: {X.shape[0]}")
print(f"   Features: {X.shape[1]}")
print(f"   Training set: {X_train.shape[0]} samples")
print(f"   Test set: {X_test.shape[0]} samples")
print(f"   Top 30 finishers: {y.mean():.2%}")

# Scale features for models that need it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 2: DEFINE ALL MODELS


print("\n STEP 2: DEFINING MODELS")
print("=" * 50)

models_config = {
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, None],
            'min_samples_split': [10, 20]
        },
        'use_scaled': False
    },
    'Logistic Regression': {
        'model': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
        'params': {
            'C': [0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        },
        'use_scaled': True
    },
    'XGBoost': {
        'model': XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1, verbosity=0),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [4, 6],
            'learning_rate': [0.05, 0.1],
            'scale_pos_weight': [3.8]
        },
        'use_scaled': False
    },
    'LightGBM': {
        'model': LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [4, 6],
            'learning_rate': [0.05, 0.1],
            'class_weight': ['balanced']
        },
        'use_scaled': False
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [3, 4],
            'learning_rate': [0.05, 0.1]
        },
        'use_scaled': False
    }
}

# 3: TRAIN ALL MODELS


print("\n STEP 3: TRAINING ALL MODELS")
print("=" * 50)

results = []
trained_models = {}

for model_name, config in models_config.items():
    print(f"\n📋 Training {model_name}...")
    start_time = time.time()
    
    # Select appropriate data (scaled or not)
    if config['use_scaled']:
        X_tr, X_te = X_train_scaled, X_test_scaled
    else:
        X_tr, X_te = X_train, X_test
    
    # Hyperparameter tuning
    grid_search = GridSearchCV(
        config['model'],
        config['params'],
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_tr, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    training_time = time.time() - start_time
    
    # Make predictions
    y_pred = best_model.predict(X_te)
    y_pred_proba = best_model.predict_proba(X_te)[:, 1]
    
    # Calculate metrics
    auc_score = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Cross-validation
    cv_scores = cross_val_score(best_model, X_tr, y_train, cv=5, scoring='roc_auc')
    
    # Store results
    model_results = {
        'Model': model_name,
        'ROC_AUC': auc_score,
        'CV_Mean_AUC': cv_scores.mean(),
        'CV_Std_AUC': cv_scores.std(),
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1,
        'Training_Time': training_time,
        'Best_Params': str(best_params),
        'Model_Object': best_model
    }
    
    results.append(model_results)
    trained_models[model_name] = best_model
    
    print(f"✅ {model_name} completed in {training_time:.1f}s")
    print(f"   Best CV: {best_score:.4f}, Test AUC: {auc_score:.4f}")
    print(f"   Precision: {precision:.3f}, Recall: {recall:.3f}")

# Create results dataframe
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('ROC_AUC', ascending=False)

print("\n MODEL PERFORMANCE RANKING")
print("=" * 50)
display_cols = ['Model', 'ROC_AUC', 'CV_Mean_AUC', 'Precision', 'Recall', 'F1_Score', 'Training_Time']
print(results_df[display_cols].round(4).to_string(index=False))


# STEP 4: FEATURE IMPORTANCE ANALYSIS


print("\n STEP 4: FEATURE IMPORTANCE ANALYSIS")
print("=" * 50)

# Get feature importance from tree-based models
feature_importance_data = {}

for model_name, model in trained_models.items():
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance_data[model_name] = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)

# Create feature importance visualization
if feature_importance_data:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (model_name, importance_df) in enumerate(feature_importance_data.items()):
        if idx < len(axes):
            top_features = importance_df.head(10).sort_values('importance', ascending=True)
            axes[idx].barh(range(len(top_features)), top_features['importance'], 
                          color='steelblue', alpha=0.7)
            axes[idx].set_yticks(range(len(top_features)))
            axes[idx].set_yticklabels(top_features['feature'])
            axes[idx].set_xlabel('Importance')
            axes[idx].set_title(f'{model_name}\nTop 10 Features', fontweight='bold')
            axes[idx].grid(True, alpha=0.3, axis='x')
    
    # Remove empty subplots
    for idx in range(len(feature_importance_data), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig('results/comprehensive/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()


# 5: COMPREHENSIVE MODEL COMPARISON


print("\n STEP 5: COMPREHENSIVE MODEL COMPARISON")
print("=" * 50)

# Create comprehensive comparison dashboard
fig = plt.figure(figsize=(20, 15))

# 1. Performance Ranking (Top Left)
ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
models_sorted = results_df.sort_values('ROC_AUC', ascending=True)

colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models_sorted)))
bars = ax1.barh(range(len(models_sorted)), models_sorted['ROC_AUC'], 
                color=colors, alpha=0.8, edgecolor='black')

ax1.set_yticks(range(len(models_sorted)))
ax1.set_yticklabels(models_sorted['Model'], fontsize=12, fontweight='bold')
ax1.set_xlabel('ROC AUC Score', fontsize=14, fontweight='bold')
ax1.set_title('MODEL PERFORMANCE RANKING', fontsize=16, fontweight='bold', pad=20)
ax1.grid(True, alpha=0.3, axis='x')
ax1.set_xlim([0, 1.0])

# Add values on bars
for i, (bar, auc) in enumerate(zip(bars, models_sorted['ROC_AUC'])):
    width = bar.get_width()
    ax1.text(width + 0.02, i, f'{auc:.3f}', va='center', fontsize=11, fontweight='bold')

# 2. ROC Curves (Top Right)
ax2 = plt.subplot2grid((3, 3), (0, 2))
ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier', linewidth=2)

for i, (model_name, model) in enumerate(trained_models.items()):
    if config['use_scaled']:
        X_te = X_test_scaled
    else:
        X_te = X_test
        
    y_pred_proba = model.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    ax2.plot(fpr, tpr, linewidth=2.5, 
            label=f'{model_name} (AUC = {auc_score:.3f})')

ax2.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax2.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax2.set_title('ROC CURVES COMPARISON', fontsize=14, fontweight='bold', pad=15)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# 3. Metrics Comparison (Middle Left)
ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
metrics = ['Precision', 'Recall', 'F1_Score']
x_pos = np.arange(len(results_df))
width = 0.25

for i, metric in enumerate(metrics):
    values = results_df[metric].values
    ax3.bar(x_pos + i*width, values, width, label=metric, alpha=0.8)

ax3.set_xlabel('Models', fontsize=12, fontweight='bold')
ax3.set_ylabel('Score', fontsize=12, fontweight='bold')
ax3.set_title('CLASSIFICATION METRICS', fontsize=14, fontweight='bold', pad=15)
ax3.set_xticks(x_pos + width)
ax3.set_xticklabels([name.split()[0] for name in results_df['Model']], 
                   rotation=45, ha='right')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Training Time (Middle Right)
ax4 = plt.subplot2grid((3, 3), (1, 2))
time_bars = ax4.bar(results_df['Model'], results_df['Training_Time'], alpha=0.8)
ax4.set_title('TRAINING TIME COMPARISON', fontsize=14, fontweight='bold', pad=15)
ax4.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
ax4.tick_params(axis='x', rotation=45)
ax4.grid(True, alpha=0.3)

for bar in time_bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{height:.1f}s', ha='center', va='bottom', fontweight='bold')

# 5. Precision-Recall Curves (Bottom Left)
ax5 = plt.subplot2grid((3, 3), (2, 0), colspan=2)
random_precision = y_test.mean()

for i, (model_name, model) in enumerate(trained_models.items()):
    if config['use_scaled']:
        X_te = X_test_scaled
    else:
        X_te = X_test
        
    y_pred_proba = model.predict_proba(X_te)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    ax5.plot(recall, precision, linewidth=2.5,
            label=f'{model_name} (AP = {avg_precision:.3f})')

ax5.axhline(y=random_precision, color='k', linestyle='--', alpha=0.5,
           label=f'Random (AP = {random_precision:.3f})')
ax5.set_xlabel('Recall', fontsize=12, fontweight='bold')
ax5.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax5.set_title('PRECISION-RECALL CURVES', fontsize=14, fontweight='bold', pad=15)
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Performance Summary Table (Bottom Right)
ax6 = plt.subplot2grid((3, 3), (2, 2))
ax6.axis('tight')
ax6.axis('off')

table_data = results_df[['Model', 'ROC_AUC', 'Precision', 'Recall', 'F1_Score']].round(3)
table = ax6.table(cellText=table_data.values,
                 colLabels=table_data.columns,
                 cellLoc='center',
                 loc='center',
                 bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.8)

plt.tight_layout()
plt.savefig('results/comprehensive/comprehensive_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 6: DETAILED BEST MODEL ANALYSIS


print("\n STEP 6: DETAILED BEST MODEL ANALYSIS")
print("=" * 50)

best_model_name = results_df.iloc[0]['Model']
best_model = results_df.iloc[0]['Model_Object']
best_auc = results_df.iloc[0]['ROC_AUC']

print(f" BEST PERFORMING MODEL: {best_model_name}")
print(f" Test AUC: {best_auc:.4f}")

# Detailed evaluation of best model
if best_model_name in [config for config in models_config if models_config[config]['use_scaled']]:
    X_te = X_test_scaled
else:
    X_te = X_test

y_pred_best = best_model.predict(X_te)
y_pred_proba_best = best_model.predict_proba(X_te)[:, 1]

print(f"\n📋 Detailed Classification Report:")
print(classification_report(y_test, y_pred_best, target_names=['Outside Top 30', 'Top 30']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Outside Top 30', 'Top 30'],
            yticklabels=['Outside Top 30', 'Top 30'])
plt.title(f'Confusion Matrix - {best_model_name}', fontweight='bold', fontsize=14)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('results/comprehensive/best_model_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()


# STEP 7: BUSINESS INSIGHTS AND RECOMMENDATIONS


print("\n STEP 7: BUSINESS INSIGHTS AND RECOMMENDATIONS")
print("=" * 55)

print(f"\n FINAL RECOMMENDATION:")
print(f"   Use {best_model_name} for production deployment")
print(f"   Expected performance: ROC AUC = {best_auc:.3f}")



# Feature importance insights
if best_model_name in feature_importance_data:
    best_importance = feature_importance_data[best_model_name]
    print(f"\n KEY SUCCESS FACTORS ({best_model_name}):")
    print("=" * 40)
    
    top_5_features = best_importance.head(5)
    for i, row in top_5_features.iterrows():
        print(f"   {i+1}. {row['feature']}: {row['importance']:.3f}")


#  8: SAVE MODELS AND RESULTS


print("\n STEP 8: SAVING MODELS AND RESULTS")
print("=" * 50)

# Save all models
for model_name, model in trained_models.items():
    filename = f"models/{model_name.lower().replace(' ', '_')}.pkl"
    joblib.dump(model, filename)
    print(f"✅ Saved {model_name}")

# Save best model separately
joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

# Save comprehensive results
results_df.to_csv('results/comprehensive/model_performance.csv', index=False)

# Save feature importance
if feature_importance_data:
    for model_name, importance_df in feature_importance_data.items():
        filename = f"results/comprehensive/feature_importance_{model_name.lower().replace(' ', '_')}.csv"
        importance_df.to_csv(filename, index=False)

# Create final summary report
summary_report = f"""
PROJECT SUMMARY
====================

PROJECT: Cycling Race Top 30 Finish Prediction
DATASET: {X.shape[0]} samples, {X.shape[1]} features
BEST MODEL: {best_model_name}
PERFORMANCE: ROC AUC = {best_auc:.3f}

MODEL RANKING:
"""
for i, row in results_df.iterrows():
    summary_report += f"{i+1}. {row['Model']}: AUC = {row['ROC_AUC']:.3f}, F1 = {row['F1_Score']:.3f}\n"




with open('results/comprehensive/project_summary.txt', 'w') as f:
    f.write(summary_report)

print(f"\n FILES SAVED:")
print("   - models/ (all trained models)")
print("   - results/comprehensive/ (all analysis results)")
print("   - results/comprehensive/project_summary.txt")

print("\n" + "="*70)
print("COMPLETE CYCLING PREDICTION ANALYSIS")
print("="*70)

print(f"\n PROJECT COMPLETED SUCCESSFULLY")
print(f"   Best Model: {best_model_name}")
print(f"   Performance: ROC AUC = {best_auc:.3f}")
print(f"   Ready for deployment!") 