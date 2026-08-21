import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import xgboost as xgb

def train_and_evaluate(master_dataset_path="data/processed/master_risk_dataset.csv",
                       model_dir="models",
                       n_splits=5,
                       seed=42):
    """
    Evaluates ML models using 5-Fold Stratified Cross-Validation, computing Mean ± Std Dev,
    per-class precision/recall/F1 breakdowns, and multi-class confusion matrices across Low, Medium, High, Critical classes.
    """
    print(f"[Model Evaluator] Loading master dataset from: {master_dataset_path}")
    os.makedirs(model_dir, exist_ok=True)

    df = pd.read_csv(master_dataset_path)

    # Exclude non-feature metadata columns
    non_feature_cols = ["Zone_ID", "Latitude", "Longitude", "Land_Cover", "Risk_Score", "Risk_Level"]
    feature_cols = [c for c in df.columns if c not in non_feature_cols]

    X = df[feature_cols].copy()
    
    # Target encoding: Low -> 0, Medium -> 1, High -> 2, Critical -> 3
    target_map = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
    inv_target_map = {v: k for k, v in target_map.items()}
    y = df["Risk_Level"].map(target_map)

    classes = sorted(list(np.unique(y)))
    class_names = [inv_target_map[c] for c in classes]

    print(f"[Model Evaluator] Tabular dataset shape: X={X.shape}, y={y.shape}")
    print(f"[Model Evaluator] Class distribution: {dict(df['Risk_Level'].value_counts())}")

    models = {
        "Logistic Regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed)),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=100, max_depth=8, random_state=seed),
        "XGBoost": lambda: xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=seed, eval_metric="mlogloss")
    }

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    summary_results = []
    confusion_matrices_dict = {}
    per_class_reports_dict = {}

    print("\n" + "="*90)
    print(f"            5-FOLD STRATIFIED CROSS-VALIDATION EVALUATION RESULTS")
    print("="*90)

    for name, model_fn in models.items():
        accs, precs, recs, f1s, aucs = [], [], [], [], []
        cum_cm = np.zeros((len(classes), len(classes)), dtype=int)
        y_true_all, y_pred_all = [], []

        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            clf = model_fn()
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_val)

            accs.append(accuracy_score(y_val, y_pred))
            precs.append(precision_score(y_val, y_pred, average="weighted", zero_division=0))
            recs.append(recall_score(y_val, y_pred, average="weighted", zero_division=0))
            f1s.append(f1_score(y_val, y_pred, average="weighted", zero_division=0))

            y_true_all.extend(y_val)
            y_pred_all.extend(y_pred)

            if hasattr(clf, "predict_proba"):
                try:
                    y_proba = clf.predict_proba(X_val)
                    if len(np.unique(y_val)) > 1:
                        auc_val = roc_auc_score(y_val, y_proba, multi_class="ovr", average="weighted")
                        aucs.append(auc_val)
                except Exception:
                    pass

            cm = confusion_matrix(y_val, y_pred, labels=classes)
            cum_cm += cm

        # Calculate Mean ± Std
        m_acc, s_acc = np.mean(accs), np.std(accs)
        m_prec, s_prec = np.mean(precs), np.std(precs)
        m_rec, s_rec = np.mean(recs), np.std(recs)
        m_f1, s_f1 = np.mean(f1s), np.std(f1s)
        m_auc, s_auc = (np.mean(aucs), np.std(aucs)) if aucs else (float("nan"), float("nan"))

        summary_results.append({
            "Model": name,
            "Accuracy": f"{m_acc:.4f} ± {s_acc:.4f}",
            "Precision": f"{m_prec:.4f} ± {s_prec:.4f}",
            "Recall": f"{m_rec:.4f} ± {s_rec:.4f}",
            "F1-Score": f"{m_f1:.4f} ± {s_f1:.4f}",
            "ROC-AUC": f"{m_auc:.4f} ± {s_auc:.4f}" if not np.isnan(m_auc) else "N/A",
            "Accuracy_Mean": m_acc,
            "F1_Mean": m_f1
        })

        confusion_matrices_dict[name] = {
            "classes": class_names,
            "matrix": cum_cm.tolist()
        }

        # Per-class breakdown report
        report_dict = classification_report(y_true_all, y_pred_all, target_names=class_names, output_dict=True, zero_division=0)
        per_class_reports_dict[name] = report_dict

    results_df = pd.DataFrame(summary_results)
    display_df = results_df[["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]]
    print(display_df.to_string(index=False))
    print("="*90 + "\n")

    # Save final Logistic Regression baseline & Random Forest model artifacts
    rf_final = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=seed)
    rf_final.fit(X, y)

    rf_save_path = os.path.join(model_dir, "random_forest_model.joblib")
    joblib.dump({"model": rf_final, "feature_names": feature_cols, "target_map": target_map}, rf_save_path)
    print(f"[Model Evaluator] Saved final Random Forest model to: {rf_save_path}")

    # Save metrics table
    results_df.to_csv(os.path.join(model_dir, "model_comparison_metrics.csv"), index=False)
    
    # Save confusion matrices
    cm_path = os.path.join(model_dir, "confusion_matrices.json")
    with open(cm_path, "w") as f:
        json.dump(confusion_matrices_dict, f, indent=2)

    # Save per-class classification reports
    rep_path = os.path.join(model_dir, "per_class_classification_reports.json")
    with open(rep_path, "w") as f:
        json.dump(per_class_reports_dict, f, indent=2)

    print(f"[Model Evaluator] Saved confusion matrices & per-class reports to: {model_dir}/")
    return rf_save_path, results_df

if __name__ == "__main__":
    train_and_evaluate()
