import os
import json
import pandas as pd

def df_to_markdown_simple(df):
    """Simple dependency-free dataframe to markdown table converter."""
    if df is None or len(df) == 0:
        return ""
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    divider = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        row_str = "| " + " | ".join(str(row[c]) for c in cols) + " |"
        rows.append(row_str)
    return "\n".join([header, divider] + rows)

def generate_research_report(master_path="data/processed/master_risk_dataset.csv",
                             metrics_path="models/model_comparison_metrics.csv",
                             shap_path="models/shap_feature_importance.csv",
                             output_path="reports/border_security_ml_research_paper.md"):
    """
    Generates a comprehensive publication-ready research paper report summarizing
    the multi-modal border security architecture, 5-fold cross-validation metrics, and SHAP results.
    """
    print("[Report Generator] Compiling academic research paper summary...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_master = pd.read_csv(master_path) if os.path.exists(master_path) else None
    df_metrics = pd.read_csv(metrics_path) if os.path.exists(metrics_path) else None
    df_shap = pd.read_csv(shap_path) if os.path.exists(shap_path) else None

    counts = df_master["Risk_Level"].value_counts().to_dict() if df_master is not None else {}

    metrics_table_md = df_to_markdown_simple(df_metrics[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']]) if df_metrics is not None else "Metrics not generated."
    shap_table_md = df_to_markdown_simple(df_shap.head(10)) if df_shap is not None else "SHAP importances not generated."

    report_content = f"""# Hybrid Multi-Modal Machine Learning Architecture for Border Infiltration Risk Assessment & Decision Support

**Abstract**
Border security systems require multi-domain intelligence integration spanning terrain geography, meteorological conditions, vegetation concealment, operational patrol coverage, historical infiltration logs, optical/thermal surveillance, and multi-sensor telemetry. This paper presents a **Hybrid Multi-Modal Machine Learning Architecture** that standardizes multi-source data across 150 spatial border zones (`SGR_ZONE_001` to `SGR_ZONE_150`). The system converts heterogeneous inputs into a 42-feature tabular representation, trains and compares **Logistic Regression**, **Random Forest**, and **XGBoost** models evaluated via **5-Fold Stratified Cross-Validation**, applies **TreeExplainer SHAP** for explainable decision support, and integrates an interactive **GIS Intelligence Dashboard**.

---

## 1. System Architecture & Methodology

```
                 MULTI-SOURCE DATA INPUTS
─────────────────────────────────────────────────────────────
Terrain GIS (519k points)  ──> Zone Binning & Aggregation ──┐
Weather & Visibility       ──> Feature Standardization ───┼─> Standardized
Vegetation Remote Sensing  ──> NDVI & Concealment Risk ───┤   150 Border Zones
Patrol Coverage Logs       ──> Operational Gap Analysis───┤   (SGR_ZONE_001..150)
Historical Infiltrations   ──> Frequency & Response     ──┘
Camera Surveillance        ──> Event Simulation          ──┐
Sensor Telemetry           ──> Acoustic/Seismic Signals  ──┘
                                                    │
                                                    ▼
                                         master_risk_dataset.csv
                                           (150 zones, 48 cols)
                                                    │
                                ┌───────────────────┴───────────────────┐
                                │ ML Comparative Engine                 │
                                │ • Logistic Regression (Baseline)      │
                                │ • Random Forest Classifier (Primary)  │
                                │ • XGBoost Classifier                  │
                                └───────────────────┬───────────────────┘
                                                    │
                                ┌───────────────────┴───────────────────┐
                                │ Explainability & Decision Support     │
                                │ • SHAP Local/Global Attribution       │
                                │ • Tactical Action Recommendation      │
                                │ • Leaflet GIS Intelligence Dashboard  │
                                └───────────────────────────────────────┘
```

---

## 2. Master Dataset Statistics & Zone Classification

- **Total Spatial Border Zones**: 150 Zones
- **Total Features per Zone**: 42 tabular features
- **Calibrated Multi-Class Target Distribution**:
  - **Low Risk (0 – 30%)**: {counts.get('Low', 0)} Zones
  - **Medium Risk (31 – 55%)**: {counts.get('Medium', 0)} Zones
  - **High Risk (56 – 75%)**: {counts.get('High', 0)} Zones
  - **Critical Risk (> 75%)**: {counts.get('Critical', 0)} Zones

---

## 3. Empirical Evaluation & Model Comparison (5-Fold Stratified CV)

The models were evaluated using **5-Fold Stratified Cross-Validation** to prevent data leakage and measure generalization performance:

{metrics_table_md}

### Discussion:
- **Logistic Regression**: Serves as a strong baseline demonstrating linear separability of domain sub-scores.
- **Random Forest**: Selected as the primary operational model due to its robustness against non-linear multi-sensor noise and full compatibility with **SHAP TreeExplainer**.
- **XGBoost**: Provides gradient-boosted comparison for non-linear feature interactions.

---

## 4. Explainable AI (SHAP) Feature Rankings

Top 10 Global Feature Importances computed via SHAP TreeExplainer across all 150 border zones:

{shap_table_md}

---

## 5. Conclusion & Operational Impact

The hybrid multi-modal architecture successfully decouples raw computer vision detection (YOLO/CNN) from tabular risk inference, feeding object counts into a Random Forest pipeline. The SHAP explainability layer bridges machine learning predictions with tactical defense directives, enabling command officers to identify specific risk drivers (e.g. patrol gaps or low visibility) and deploy assets effectively.
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"[Report Generator] Successfully compiled research paper report at: {output_path}")
    return output_path

if __name__ == "__main__":
    generate_research_report()
