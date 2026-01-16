import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# load data
train_file_path = "../Data/training.csv"
save_trained_file = "./Models/xgb_overtake_model.joblib"

df = pd.read_csv(train_file_path)
required_cols = ["lat", "lon", "Left", "Right", "Confirmed"]
assert all(col in df.columns for col in required_cols), "Train CSV missing required columns!"

# convert to binary classification
print("Initial target value distribution:")
print(df["Confirmed"].value_counts())
print(f"Unique values: {sorted(df['Confirmed'].unique())}")

# convert to binary: 0 = no overtake, 1 = overtake
df["Confirmed"] = (df["Confirmed"] > 0).astype(int)

print("\nAfter converting to binary:")
print(f"Class distribution:\n{df['Confirmed'].value_counts()}")
print(f"0 (No overtake): {(df['Confirmed'] == 0).sum()}")
print(f"1 (Overtake): {(df['Confirmed'] == 1).sum()}")
print(f"Class ratio: {(df['Confirmed'] == 0).sum() / (df['Confirmed'] == 1).sum():.2f}:1")

# check if we have enough samples for both classes
if df["Confirmed"].nunique() < 2:
    raise ValueError("Need at least 2 classes for binary classification!")
if min(df["Confirmed"].value_counts()) < 2:
    print("Warning: One class has very few samples. Consider collecting more data or using different evaluation.")

# feature engineering
df["left_minus_right"] = df["Left"] - df["Right"]
df["left_plus_right"] = df["Left"] + df["Right"]
df["left_div_right"] = df["Left"] / (df["Right"] + 1e-8)

df["distance"] = np.sqrt(df["lat"] ** 2 + df["lon"] ** 2)  # Distance from origin
df["left_right_ratio"] = df["Left"] / (df["Right"] + 1e-8)
df["lat_lon_product"] = df["lat"] * df["lon"]

FEATURES = [
    "lat", "lon", "Left", "Right",
    "left_minus_right", "left_plus_right", "left_div_right",
    "distance", "left_right_ratio", "lat_lon_product"
]
TARGET = "Confirmed"

X = df[FEATURES]
y = df[TARGET]

# train-validation split with stratification
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Training class distribution: {Counter(y_train)}")
print(f"Validation class distribution: {Counter(y_val)}")


# parameter tuning
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 3.0),
        "objective": "binary:logistic",
        "random_state": 42,
        "eval_metric": "auc",
        "early_stopping_rounds": 50  # ✅ Add here
    }

    model = xgb.XGBClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    y_pred_prob = model.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, y_pred_prob)
    return score


# Run optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30, show_progress_bar=True)

print("\nBest Params:", study.best_params)

# Train final model
best_params = study.best_params
best_params.update({
    "objective": "binary:logistic",
    "random_state": 42,
    "eval_metric": "auc",
    "early_stopping_rounds": 50  # ✅ Add here
})

model = xgb.XGBClassifier(**best_params)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)

# model evaluation
print("\n" + "=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

# predictions
y_val_pred = model.predict(X_val)
y_val_prob = model.predict_proba(X_val)[:, 1]

# classification report
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_val, y_val_pred, target_names=['No Overtake', 'Overtake']))

# matrix
print("\n=== CONFUSION MATRIX ===")
cm = confusion_matrix(y_val, y_val_pred)
print(cm)

# Plot matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Overtake', 'Overtake'],
            yticklabels=['No Overtake', 'Overtake'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig("../Output/confusion_matrix.png", dpi=300)
plt.close()

# ROC-AUC Score
roc_auc = roc_auc_score(y_val, y_val_prob)
print(f"\n=== ROC-AUC SCORE ===")
print(f"ROC-AUC: {roc_auc:.4f}")

print("\n=== FEATURE IMPORTANCE ===")
feature_importance = pd.DataFrame({
    'Feature': FEATURES,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance - XGBoost')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=300)
plt.close()

# save model and metadata
model_artifact = {
    'model': model,
    'features': FEATURES,
    'feature_importance': feature_importance,
    'best_params': best_params,
    'roc_auc': roc_auc,
    'data_shape': {
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'features': len(FEATURES)
    },
    'class_distribution': {
        'training': dict(Counter(y_train)),
        'validation': dict(Counter(y_val))
    }
}
# save model data
joblib.dump(model_artifact, save_trained_file)
print(f"\nModel saved to: {save_trained_file}")

# cross validation
if len(X_train) > 1000:
    print("\n" + "=" * 50)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 50)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_cv_train, X_cv_val = X.iloc[train_idx], X.iloc[val_idx]
        y_cv_train, y_cv_val = y.iloc[train_idx], y.iloc[val_idx]

        fold_model = xgb.XGBClassifier(**best_params)
        fold_model.fit(
            X_cv_train, y_cv_train,
            eval_set=[(X_cv_val, y_cv_val)],
            verbose=False,
            early_stopping_rounds=50
        )

        # Score
        y_cv_pred_prob = fold_model.predict_proba(X_cv_val)[:, 1]
        fold_score = roc_auc_score(y_cv_val, y_cv_pred_prob)
        cv_scores.append(fold_score)

        print(f"Fold {fold}: ROC-AUC = {fold_score:.4f}")

    print(f"\nCV Mean ROC-AUC: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

print("\n" + "=" * 50)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 50)