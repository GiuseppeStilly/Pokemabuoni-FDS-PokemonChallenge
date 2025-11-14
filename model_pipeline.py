import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Import ML Libraries ---
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    cross_val_predict,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
from sklearn.calibration import CalibratedClassifierCV

# --- Suppress Warnings ---
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# Helper function for model tuning
def tune_model(name, estimator, param_grid, X_train_val, y_train_val, cv_strategy):
    """
    Internal helper function to run GridSearchCV and print detailed results.
    """
    print(f"--- Starting tuning for: {name} ---")

    # Calculate number of combinations
    n_combinations = 1
    for k in param_grid:
        n_combinations *= len(param_grid[k])
    print(f"(Will test {n_combinations * cv_strategy.n_splits} models)")

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring="accuracy",
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train_val, y_train_val)

    # Get the full results
    results = grid_search.cv_results_
    best_index = grid_search.best_index_

    # Extract the individual fold scores
    fold_scores = []
    for i in range(cv_strategy.n_splits):
        fold_key = f"split{i}_test_score"
        score = results[fold_key][best_index]
        fold_scores.append(score)

    mean_accuracy = np.mean(fold_scores)
    std_dev = np.std(fold_scores)

    print(f"\n--- Results for {name} ---")
    print(f"Best mean score: {mean_accuracy * 100:.2f}%")
    print(f"Score std dev (variance): +/- {std_dev * 100:.2f}%")
    print(f"Individual fold scores: {np.round(fold_scores, 4)}")
    print(f"Best params: {grid_search.best_params_}\n")

    return grid_search.best_estimator_


# Model 1: Logistic Regression (Elastic Net)
def run_model_1_logistic_net(X_processed, y):
    """
    Runs the full tuning and evaluation process
    for the Logistic Regression (Elastic Net) model.
    Returns the trained (fitted) model.
    """

    print("==========================================================")
    print("--- Starting Model 1: Logistic Regression (Elastic Net) ---")
    print("==========================================================")

    # --- 1. Split dataset ---
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- 2. Cross-validation strategy ---
    k_folds = 5
    cv_strategy = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    print(f"Data split. Training/Validation: {len(y_train_val)}, Test: {len(y_test)}")

    # --- 3. Define pipeline and parameter grid ---
    pipeline_lr = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model_lr",
                LogisticRegression(
                    random_state=42, solver="saga", penalty="elasticnet", max_iter=3000
                ),
            ),
        ]
    )

    param_grid_lr = {
        "model_lr__C": [0.01, 0.1, 1.0, 10.0],
        "model_lr__l1_ratio": [0.1, 0.5, 0.9],
    }

    # --- 4. Run the tuning ---
    best_lr = tune_model(
        "Logistic Regression (Elastic Net)",
        pipeline_lr,
        param_grid_lr,
        X_train_val,
        y_train_val,
        cv_strategy,
    )

    # --- 5. Final evaluation ---
    print("--- Final Evaluation of Model 1 ---")
    y_pred_train = best_lr.predict(X_train_val)
    accuracy_train = accuracy_score(y_train_val, y_pred_train)
    y_pred_test = best_lr.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    print(f"Accuracy (Train):    {accuracy_train * 100:.2f}%")
    print(f"Accuracy (Test):     {accuracy_test * 100:.2f}%")
    print(f"Overfitting (Gap):   {(accuracy_train - accuracy_test) * 100:.2f}%")

    print("\nClassification report on test set")
    print(
        classification_report(y_test, y_pred_test, target_names=["loss (0)", "win (1)"])
    )

    # --- 6. Return the trained model ---
    print("--- Model 1 (Logistic Net) Training Complete ---")

    return best_lr


# Model 2: Calibrated Stacking (LR, SVM, XGB)
def run_model_2_calibrated_stacking(X_processed, y):
    """
    Runs the full tuning and training process
    for the Calibrated Stacking model (LR, SVM, XGB).
    Returns a dictionary of the trained models.
    """

    print("==========================================================")
    print("--- Starting Model 2: Calibrated Stacking (LR, SVM, XGB) ---")
    print("==========================================================")

    # --- 1. Split dataset ---
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- 2. Cross-validation strategy ---
    k_folds = 5
    cv_strategy = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    print(f"Data split. Training/Validation: {len(y_train_val)}, Test: {len(y_test)}")

    # --- 3. Define pipelines and parameter grids (Level 1 Experts) ---
    pipeline_lr = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model_lr",
                LogisticRegression(random_state=42, solver="saga", max_iter=3000),
            ),
        ]
    )
    param_grid_lr = {
        "model_lr__penalty": ["elasticnet"],
        "model_lr__C": [0.1, 1.0, 10.0],
        "model_lr__l1_ratio": [0.1, 0.5, 0.9],
    }

    pipeline_svm = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model_svm", SVC(random_state=42, probability=True)),
        ]
    )
    param_grid_svm = {
        "model_svm__C": [0.1, 1.0, 10.0],
        "model_svm__kernel": ["rbf", "linear"],
    }

    pipeline_xgb = Pipeline(
        steps=[
            (
                "model_xgb",
                XGBClassifier(
                    random_state=42,
                    n_jobs=-1,
                    use_label_encoder=False,
                    eval_metric="logloss",
                ),
            )
        ]
    )
    param_grid_xgb = {
        "model_xgb__n_estimators": [100, 200, 400],
        "model_xgb__max_depth": [4, 6, 7],
        "model_xgb__learning_rate": [0.05, 0.1],
    }

    # --- 4. Tune the experts ---
    best_lr = tune_model(
        "LR (Stack)", pipeline_lr, param_grid_lr, X_train_val, y_train_val, cv_strategy
    )
    best_svm = tune_model(
        "SVM (Stack)",
        pipeline_svm,
        param_grid_svm,
        X_train_val,
        y_train_val,
        cv_strategy,
    )
    best_xgb = tune_model(
        "XGB (Stack)",
        pipeline_xgb,
        param_grid_xgb,
        X_train_val,
        y_train_val,
        cv_strategy,
    )

    # --- 5. CALIBRATE THE EXPERTS ---
    print("\n--- Calibrating uncalibrated models (SVM, XGBoost) ---")
    calibrated_svm = CalibratedClassifierCV(
        best_svm, method="isotonic", cv=cv_strategy, n_jobs=-1
    )
    calibrated_xgb = CalibratedClassifierCV(
        best_xgb, method="isotonic", cv=cv_strategy, n_jobs=-1
    )

    print("Fitting SVM calibrator...")
    calibrated_svm.fit(X_train_val, y_train_val)
    print("Fitting XGBoost calibrator...")
    calibrated_xgb.fit(X_train_val, y_train_val)
    print("...Calibration completed.")

    # --- 6. CREATE META-FEATURES ---
    print("\n--- Creating Meta-Features (from Calibrated models) ---")
    meta_preds_lr = cross_val_predict(
        best_lr,
        X_train_val,
        y_train_val,
        cv=cv_strategy,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    meta_preds_svm_cal = cross_val_predict(
        calibrated_svm,
        X_train_val,
        y_train_val,
        cv=cv_strategy,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    meta_preds_xgb_cal = cross_val_predict(
        calibrated_xgb,
        X_train_val,
        y_train_val,
        cv=cv_strategy,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    meta_X_train = np.column_stack(
        (meta_preds_lr, meta_preds_svm_cal, meta_preds_xgb_cal)
    )

    # --- 7. TRAIN META-MODEL ("Manager") ---
    print("\n--- Training Meta-Model (Manager) ---")
    meta_model = LogisticRegression(random_state=42)
    meta_model.fit(meta_X_train, y_train_val)

    coefs = meta_model.coef_[0]
    print(f"Meta-Model Coefficients (Learned Weights):")
    print(f"  [LR: {coefs[0]:.4f}, SVM_Cal: {coefs[1]:.4f}, XGB_Cal: {coefs[2]:.4f}]")

    # --- 8. FINAL EVALUATION ---
    print("\n--- Final Evaluation of Stacking Model on Test Set ---")
    preds_test_lr = best_lr.predict_proba(X_test)[:, 1]
    preds_test_svm_cal = calibrated_svm.predict_proba(X_test)[:, 1]
    preds_test_xgb_cal = calibrated_xgb.predict_proba(X_test)[:, 1]

    meta_X_test = np.column_stack(
        (preds_test_lr, preds_test_svm_cal, preds_test_xgb_cal)
    )

    final_predictions = meta_model.predict(meta_X_test)
    final_accuracy = accuracy_score(y_test, final_predictions)
    final_predictions_train = meta_model.predict(meta_X_train)
    accuracy_train = accuracy_score(y_train_val, final_predictions_train)

    print("\n--- Mixture of Experts (Calibrated Stacking) Results ---")
    print(f"Accuracy (Train):    {accuracy_train * 100:.2f}%")
    print(f"Accuracy (Test):     {final_accuracy * 100:.2f}%")
    print(f"Overfitting (Gap):   {(accuracy_train - final_accuracy) * 100:.2f}%")
    print("\nClassification report on test set")
    print(
        classification_report(
            y_test, final_predictions, target_names=["loss (0)", "win (1)"]
        )
    )

    # --- 9. Return the models for submission ---
    print("--- Model 2 (Calibrated Stacking) Training Complete ---")
    return {
        "meta_model": meta_model,
        "best_lr": best_lr,
        "calibrated_svm": calibrated_svm,
        "calibrated_xgb": calibrated_xgb,
    }


# Model 3: Calibrated Voting (LR, SVM, LGBM)
def run_model_3_calibrated_voting(X_processed, y):
    """
    Runs the full tuning and training process
    for the Calibrated VotingClassifier (LR, SVM, LGBM).
    Returns the trained (fitted) model.
    """

    print("==========================================================")
    print("--- Starting Model 3: Calibrated Voting (LR, SVM, LGBM) ---")
    print("==========================================================")

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print(f"Data split. Training/Validation: {len(y_train_val)}, Test: {len(y_test)}")

    # Define pipelines and parameter grids
    pipeline_lr = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model_lr",
                LogisticRegression(random_state=42, solver="saga", max_iter=3000),
            ),
        ]
    )
    param_grid_lr = {
        "model_lr__penalty": ["elasticnet"],
        "model_lr__C": [0.1, 1.0, 10.0],
        "model_lr__l1_ratio": [0.1, 0.5, 0.9],
    }
    pipeline_svm = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model_svm", SVC(random_state=42, probability=True)),
        ]
    )
    param_grid_svm = {
        "model_svm__C": [0.1, 1.0, 10.0],
        "model_svm__kernel": ["rbf", "linear"],
    }
    pipeline_lgbm = Pipeline(
        steps=[("model_lgbm", LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1))]
    )
    param_grid_lgbm = {
        "model_lgbm__n_estimators": [100, 200],
        "model_lgbm__max_depth": [4, 6],
        "model_lgbm__learning_rate": [0.1],
    }

    # Tune models
    best_lr = tune_model(
        "LR (Voting)", pipeline_lr, param_grid_lr, X_train_val, y_train_val, cv_strategy
    )
    best_svm = tune_model(
        "SVM (Voting)",
        pipeline_svm,
        param_grid_svm,
        X_train_val,
        y_train_val,
        cv_strategy,
    )
    best_lgbm = tune_model(
        "LGBM (Voting)",
        pipeline_lgbm,
        param_grid_lgbm,
        X_train_val,
        y_train_val,
        cv_strategy,
    )

    # Calibrate
    print("\n--- Calibrating uncalibrated models (SVM, LGBM) ---")
    calibrated_svm = CalibratedClassifierCV(
        best_svm, method="isotonic", cv=cv_strategy, n_jobs=-1
    )
    calibrated_lgbm = CalibratedClassifierCV(
        best_lgbm, method="isotonic", cv=cv_strategy, n_jobs=-1
    )
    print("Fitting SVM calibrator...")
    calibrated_svm.fit(X_train_val, y_train_val)
    print("Fitting LGBM calibrator...")
    calibrated_lgbm.fit(X_train_val, y_train_val)
    print("...Calibration completed.")

    # Training Final VotingClassifier
    print("\n--- Training Final VotingClassifier (with Calibrated Experts) ---")
    estimators = [
        ("lr", best_lr),
        ("svm_cal", calibrated_svm),
        ("lgbm_cal", calibrated_lgbm),
    ]
    voting_clf = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
    voting_clf.fit(X_train_val, y_train_val)
    print("...Training completed.\n")

    # Final evaluation
    print("--- Final Evaluation of VotingClassifier on Test Set ---")
    y_pred_train = voting_clf.predict(X_train_val)
    accuracy_train = accuracy_score(y_train_val, y_pred_train)
    y_pred_test = voting_clf.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    print(f"Accuracy (Train):    {accuracy_train * 100:.2f}%")
    print(f"Accuracy (Test):     {accuracy_test * 100:.2f}%")
    print(f"Overfitting (Gap):   {(accuracy_train - accuracy_test) * 100:.2f}%")

    print("--- Model 3 (Calibrated Voting) Training Complete ---")

    return voting_clf
