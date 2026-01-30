import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, \
    precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
from sklearn.metrics import roc_curve


warnings.filterwarnings('ignore')

# load data
train_file_path = "../Data/training.csv"
save_trained_file = "../Model/rf_overtake_model.joblib"

df = pd.read_csv(train_file_path)
required_cols = ["lat", "lon", "Left", "Right", "Confirmed"]
assert all(col in df.columns for col in required_cols), "missing required columns!"

# to binary classification
print("first target value distribution:")
print(df["Confirmed"].value_counts())
print(f"Unique values: {sorted(df['Confirmed'].unique())}")

# to binary: 0 for no overtake 1 fro overtake
df["Confirmed"] = (df["Confirmed"] > 0).astype(int)

print("\nconverting to binary:")
print(f"class distribution:\n{df['Confirmed'].value_counts()}")
print(f"0 (no overtake): {(df['Confirmed'] == 0).sum()}")
print(f"1 (overtake): {(df['Confirmed'] == 1).sum()}")
class_ratio = (df['Confirmed'] == 0).sum() / (df['Confirmed'] == 1).sum()
print(f"class ratio: {class_ratio:.2f}:1")

# calc class weights for imbalance handling
classes = np.array([0, 1])
class_weights = compute_class_weight('balanced', classes=classes, y=df["Confirmed"])
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"class weights: {class_weight_dict}")

# to check if we have enough samples for both classes
if df["Confirmed"].nunique() < 2:
    raise ValueError("need at least 2 classes for binary classification!")
if min(df["Confirmed"].value_counts()) < 2:
    print("warning: one class has very few samples")

# Feature engineering
df["left_minus_right"] = df["Left"] - df["Right"]
df["left_plus_right"] = df["Left"] + df["Right"]
df["left_div_right"] = df["Left"] / (df["Right"] + 1e-8)
df["distance"] = np.sqrt(df["lat"] ** 2 + df["lon"] ** 2)
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

# Train-validation split with stratification
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\ntraining set: {X_train.shape[0]} samples")
print(f"validation set: {X_val.shape[0]} samples")
print(f"training class distribution: {Counter(y_train)}")
print(f"validation class distribution: {Counter(y_val)}")


# Parameter tuning for Random Forest
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_categorical("max_depth", [None, 5, 10, 15, 20, 30]),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "class_weight": trial.suggest_categorical("class_weight",
                                                  ["balanced", "balanced_subsample", class_weight_dict]),
        "random_state": 42,
        # to use all available CPU cores
        "n_jobs": -1
    }

    # check max depth
    if params["max_depth"] is None:
        del params["max_depth"]

    model = RandomForestClassifier(**params)

    # Train model
    model.fit(X_train, y_train)

    # Predict probabilities for validation set
    y_val_prob = model.predict_proba(X_val)[:, 1]

    # Use Average Precision Score (better for imbalanced data)
    score = average_precision_score(y_val, y_val_prob)
    return score


# Run optimization
print("\nfro parameter optimization")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("\nParams:", study.best_params)
print(f"precision score: {study.best_value:.4f}")

# Train final model
best_params = study.best_params
best_params.update({
    "random_state": 42,
    "n_jobs": -1
})

# check for class_weight
if "class_weight" not in best_params:
    best_params["class_weight"] = class_weight_dict

model = RandomForestClassifier(**best_params)

print("\ntraining final model...")
model.fit(X_train, y_train)

# Model evaluation
print("model evaluation")

# Predictions
y_val_pred = model.predict(X_val)
y_val_prob = model.predict_proba(X_val)[:, 1]

# Classification report
print("\nreportt")
print(classification_report(y_val, y_val_pred, target_names=['No Overtake', 'Overtake']))

# confusion Matrix
print("\ncpnfusion matrix:")
cm = confusion_matrix(y_val, y_val_pred)
print(cm)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Overtake', 'Overtake'],
            yticklabels=['No Overtake', 'Overtake'])
plt.title('Random Forest - Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig("../Output/rf_confusion_matrix.png", dpi=300)
plt.close()

# ROC-AUC Score
roc_auc = roc_auc_score(y_val, y_val_prob)
print(f"\nROC-AUC score: {roc_auc:.4f}")

# avg precision score
avg_precision = average_precision_score(y_val, y_val_prob)
print(f"avg precision score: {avg_precision:.4f}")

# Plottting curve
plt.figure(figsize=(8, 6))
precision, recall, _ = precision_recall_curve(y_val, y_val_prob)
plt.plot(recall, precision, marker='.', label=f'Random Forest (AP={avg_precision:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../Output/rf_precision_recall_curve.png", dpi=300)
plt.close()

feature_importance = pd.DataFrame({
    'Feature': FEATURES,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nfeature imp:")
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Random Forest - Feature Importance')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig("../Output/rf_feature_importance.png", dpi=300)
plt.close()

# Save model and other datas
model_artifact = {
    'model': model,
    'features': FEATURES,
    'feature_importance': feature_importance,
    'best_params': best_params,
    'roc_auc': roc_auc,
    'average_precision': avg_precision,
    'data_shape': {
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'features': len(FEATURES)
    },
    'class_distribution': {
        'training': dict(Counter(y_train)),
        'validation': dict(Counter(y_val))
    },
    'class_weights': class_weight_dict
}

joblib.dump(model_artifact, save_trained_file)
print(f"\nmodel saved to: {save_trained_file}")

# Cross validation with parameters
if len(X) > 1000:
    print("cross validation with params:")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores_auc = []
    cv_scores_ap = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        # new iteration
        X_cv_train, X_cv_val = X.iloc[train_idx], X.iloc[val_idx]
        y_cv_train, y_cv_val = y.iloc[train_idx], y.iloc[val_idx]

        fold_model = RandomForestClassifier(**best_params)
        fold_model.fit(X_cv_train, y_cv_train)

        # get predictions
        y_cv_prob = fold_model.predict_proba(X_cv_val)[:, 1]

        # calculate score
        fold_score_auc = roc_auc_score(y_cv_val, y_cv_prob)
        fold_score_ap = average_precision_score(y_cv_val, y_cv_prob)

        cv_scores_auc.append(fold_score_auc)
        cv_scores_ap.append(fold_score_ap)

        print(f"Fold {fold}: ROC-AUC = {fold_score_auc:.4f}, avg precision = {fold_score_ap:.4f}")

    print(f"\nCV mean ROC-AUC: {np.mean(cv_scores_auc):.4f} (+/- {np.std(cv_scores_auc):.4f})")
    print(f"CV avg precision: {np.mean(cv_scores_ap):.4f} (+/- {np.std(cv_scores_ap):.4f})")

print("COMPLETED!!!")
#
# fpr, tpr, thresholds = roc_curve(y_val, y_val_prob)
# youden_index = tpr - fpr
# optimal_idx = np.argmax(youden_index)
# optimal_threshold = thresholds[optimal_idx]
#
# print(f"default threshold (0.5):")
# print(f"  positive predictions: {(y_val_prob >= 0.5).sum()}")
# print(f"  true positives: {((y_val_prob >= 0.5) & (y_val == 1)).sum()}")
# print(f"  false positives: {((y_val_prob >= 0.5) & (y_val == 0)).sum()}")
#
# print(f"\noptimal threshold ({optimal_threshold:.3f}) using Youden's J statistic:")
# y_val_pred_optimal = (y_val_prob >= optimal_threshold).astype(int)
# print(f"  positive predictions: {(y_val_prob >= optimal_threshold).sum()}")
# print(f"  true positives: {((y_val_prob >= optimal_threshold) & (y_val == 1)).sum()}")
# print(f"  false positives: {((y_val_prob >= optimal_threshold) & (y_val == 0)).sum()}")
#
# print("\nclassification report with optimal threshold:")
# print(classification_report(y_val, y_val_pred_optimal, target_names=['No Overtake', 'Overtake']))