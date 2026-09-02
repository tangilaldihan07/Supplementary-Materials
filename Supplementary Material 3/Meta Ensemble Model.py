# ============================================================
# TNR-WEIGHTED META-ENSEMBLE FOR BACE1 QSAR
# Consistent with the manuscript methodology
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)

from xgboost import XGBClassifier
import lightgbm as lgb

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix
)

RANDOM_STATE = 42
N_SPLITS = 5

# ============================================================
# 1. LOAD DATA
# ============================================================

train = pd.read_csv(
    r"C:\Users\Admin\Downloads\train_set.csv",
    low_memory=False
)

test = pd.read_csv(
    r"C:\Users\Admin\Downloads\test_set.csv",
    low_memory=False
)

# Clean column names
train.columns = train.columns.str.strip()
test.columns = test.columns.str.strip()

# Separate predictors and target
X_train = train.drop(columns=["Activity"])
y_train = train["Activity"]

X_test = test.drop(columns=["Activity"])
y_test = test["Activity"]


# ============================================================
# 2. ENCODE TARGET
# ============================================================

if y_train.dtype == object or y_test.dtype == object:

    le = LabelEncoder()

    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

else:

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)


# Make sure labels are binary 0/1
y_train = np.asarray(y_train).astype(int)
y_test = np.asarray(y_test).astype(int)


# ============================================================
# 3. NUMERIC CONVERSION
# ============================================================

X_train = X_train.apply(pd.to_numeric, errors="coerce")
X_test = X_test.apply(pd.to_numeric, errors="coerce")


# ============================================================
# 4. REMOVE FEATURES THAT ARE COMPLETELY MISSING
#    USING TRAINING DATA ONLY
# ============================================================

valid_columns = X_train.columns[
    ~X_train.isna().all()
]

X_train = X_train[valid_columns]
X_test = X_test.reindex(columns=valid_columns)


# ============================================================
# 5. MEAN IMPUTATION
#    FIT ONLY ON TRAINING DATA
# ============================================================

imputer = SimpleImputer(strategy="mean")

X_train = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

X_test = pd.DataFrame(
    imputer.transform(X_test),
    columns=X_train.columns,
    index=X_test.index
)


# ============================================================
# 6. REMOVE ZERO-VARIANCE FEATURES
#    USING TRAINING DATA ONLY
# ============================================================

variance = X_train.var(axis=0)

non_constant_columns = variance[variance > 0].index

X_train = X_train[non_constant_columns]
X_test = X_test[non_constant_columns]


print("Training samples :", X_train.shape[0])
print("Test samples     :", X_test.shape[0])
print("Features         :", X_train.shape[1])


# ============================================================
# 7. DEFINE FIVE BASE LEARNERS
# ============================================================

rf = RandomForestClassifier(
    random_state=RANDOM_STATE,
    n_jobs=-1
)

et = ExtraTreesClassifier(
    random_state=RANDOM_STATE,
    n_jobs=-1
)

gb = GradientBoostingClassifier(
    random_state=RANDOM_STATE
)

xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

lgbm = lgb.LGBMClassifier(
    objective="binary",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=-1
)


models = {
    "Random Forest": rf,
    "Extra Trees": et,
    "Gradient Boosting": gb,
    "XGBoost": xgb,
    "LightGBM": lgbm
}


# ============================================================
# 8. OOF TNR WEIGHT CALCULATION
#
#    IMPORTANT:
#    We calculate TNR using OUT-OF-FOLD predictions,
#    NOT training predictions.
# ============================================================

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)


def calculate_oof_tnr(model, X, y):

    oof_pred = np.zeros(len(y), dtype=int)

    for train_idx, valid_idx in skf.split(X, y):

        X_tr = X.iloc[train_idx]
        X_val = X.iloc[valid_idx]

        y_tr = y[train_idx]

        # Fresh model for each fold
        fold_model = model.__class__(
            **model.get_params()
        )

        fold_model.fit(X_tr, y_tr)

        prob = fold_model.predict_proba(X_val)[:, 1]

        pred = (prob >= 0.5).astype(int)

        oof_pred[valid_idx] = pred

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        y,
        oof_pred,
        labels=[0, 1]
    ).ravel()

    # True Negative Rate
    if (tn + fp) == 0:
        tnr = 0.0
    else:
        tnr = tn / (tn + fp)

    return tnr


# Calculate OOF TNR for each learner
oof_tnr = {}

print("\n==============================================")
print("OUT-OF-FOLD TNR")
print("==============================================")

for name, model in models.items():

    tnr = calculate_oof_tnr(
        model,
        X_train,
        y_train
    )

    oof_tnr[name] = tnr

    print(f"{name:25s}: {tnr:.4f}")


# ============================================================
# 9. NORMALIZE OOF-TNR TO OBTAIN ENSEMBLE WEIGHTS
# ============================================================

tnr_sum = sum(oof_tnr.values())

weights = {
    name: tnr / tnr_sum
    for name, tnr in oof_tnr.items()
}


print("\n==============================================")
print("NORMALIZED OOF-TNR WEIGHTS")
print("==============================================")

for name, weight in weights.items():

    print(
        f"{name:25s}: "
        f"{weight:.6f}"
    )

print(
    "\nWeight sum:",
    sum(weights.values())
)


# ============================================================
# 10. HYPERPARAMETER OPTIMIZATION
#
#     IMPORTANT:
#     The independent test set is NOT used here.
# ============================================================

param_grids = {

    "Random Forest": {

        "n_estimators": [100, 200, 300],

        "max_depth": [None, 10, 20],

        "min_samples_split": [2, 5],

        "min_samples_leaf": [1, 2]

    },

    "Extra Trees": {

        "n_estimators": [100, 200, 300],

        "max_depth": [None, 10, 20],

        "min_samples_split": [2, 5],

        "min_samples_leaf": [1, 2]

    },

    "Gradient Boosting": {

        "n_estimators": [100, 200],

        "learning_rate": [0.05, 0.1],

        "max_depth": [2, 3, 5]

    },

    "XGBoost": {

        "n_estimators": [100, 200],

        "max_depth": [3, 6],

        "learning_rate": [0.05, 0.1],

        "subsample": [0.8, 1.0],

        "colsample_bytree": [0.8, 1.0]

    },

    "LightGBM": {

        "n_estimators": [100, 200],

        "learning_rate": [0.05, 0.1],

        "num_leaves": [15, 31],

        "max_depth": [-1, 10],

        "min_child_samples": [10, 20]

    }
}


optimized_models = {}


print("\n==============================================")
print("HYPERPARAMETER OPTIMIZATION")
print("==============================================")


for name, model in models.items():

    print(f"\nOptimizing {name}...")

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grids[name],
        scoring="roc_auc",
        cv=skf,
        n_jobs=-1,
        refit=True
    )

    grid.fit(X_train, y_train)

    optimized_models[name] = grid.best_estimator_

    print("Best ROC-AUC:",
          f"{grid.best_score_:.4f}")

    print("Best parameters:")
    print(grid.best_params__)


# ============================================================
# 11. RETRAIN OPTIMIZED BASE LEARNERS ON FULL TRAINING SET
# ============================================================

print("\n==============================================")
print("FINAL MODEL TRAINING")
print("==============================================")


for name, model in optimized_models.items():

    model.fit(
        X_train,
        y_train
    )

    print(f"{name}: trained on full training set")


# ============================================================
# 12. GENERATE TEST PROBABILITIES
# ============================================================

test_probabilities = {}

for name, model in optimized_models.items():

    test_probabilities[name] = (
        model.predict_proba(X_test)[:, 1]
    )


# ============================================================
# 13. TNR-WEIGHTED SOFT VOTING
# ============================================================

ensemble_probability = np.zeros(
    len(X_test)
)

for name in optimized_models.keys():

    ensemble_probability += (
        weights[name]
        *
        test_probabilities[name]
    )


# ============================================================
# 14. CONVERT PROBABILITY TO CLASS
# ============================================================

CLASSIFICATION_THRESHOLD = 0.50

ensemble_prediction = (
    ensemble_probability >= CLASSIFICATION_THRESHOLD
).astype(int)


# ============================================================
# 15. EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    ensemble_prediction
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    ensemble_prediction
)

precision = precision_score(
    y_test,
    ensemble_prediction,
    zero_division=0
)

recall = recall_score(
    y_test,
    ensemble_prediction,
    zero_division=0
)

f1 = f1_score(
    y_test,
    ensemble_prediction,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    ensemble_probability
)

pr_auc = average_precision_score(
    y_test,
    ensemble_probability
)

mcc = matthews_corrcoef(
    y_test,
    ensemble_prediction
)


# ============================================================
# 16. CONFUSION MATRIX + TNR
# ============================================================

cm = confusion_matrix(
    y_test,
    ensemble_prediction,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

if (tn + fp) > 0:

    tnr = tn / (tn + fp)

else:

    tnr = 0.0


# ============================================================
# 17. FINAL RESULTS
# ============================================================

print("\n")
print("====================================================")
print("TNR-WEIGHTED META-ENSEMBLE RESULTS")
print("====================================================")

print(f"Accuracy          : {accuracy:.4f}")
print(f"Balanced Accuracy : {balanced_accuracy:.4f}")
print(f"Precision         : {precision:.4f}")
print(f"Recall            : {recall:.4f}")
print(f"F1-Score          : {f1:.4f}")
print(f"ROC-AUC           : {roc_auc:.4f}")
print(f"PR-AUC            : {pr_auc:.4f}")
print(f"MCC               : {mcc:.4f}")
print(f"TNR               : {tnr:.4f}")

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 18. DISPLAY ENSEMBLE WEIGHTS
# ============================================================

print("\n")
print("====================================================")
print("FINAL ENSEMBLE WEIGHTS")
print("====================================================")

weight_table = pd.DataFrame({
    "Base Learner": list(weights.keys()),
    "OOF TNR": [
        oof_tnr[name]
        for name in weights.keys()
    ],
    "Normalized Weight": [
        weights[name]
        for name in weights.keys()
    ]
})

print(weight_table.to_string(index=False))


# ============================================================
# 19. SAVE RESULTS
# ============================================================

weight_table.to_csv(
    "Meta_Ensemble_TNR_weights.csv",
    index=False
)

results_table = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Balanced Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC",
        "PR-AUC",
        "MCC",
        "TNR"
    ],

    "Value": [
        accuracy,
        balanced_accuracy,
        precision,
        recall,
        f1,
        roc_auc,
        pr_auc,
        mcc,
        tnr
    ]
})

results_table.to_csv(
    "Meta_Ensemble_Test_Results.csv",
    index=False
)

print("\nResults saved.")