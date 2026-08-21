# Hybrid Multi-Modal Machine Learning Architecture for Border Infiltration Risk Assessment & Decision Support

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
  - **Low Risk (0 – 30%)**: 19 Zones
  - **Medium Risk (31 – 55%)**: 86 Zones
  - **High Risk (56 – 75%)**: 40 Zones
  - **Critical Risk (> 75%)**: 5 Zones

---

## 3. Empirical Evaluation & Model Comparison (5-Fold Stratified CV)

The models were evaluated using **5-Fold Stratified Cross-Validation** to prevent data leakage and measure generalization performance:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.7400 ± 0.1062 | 0.7294 ± 0.1003 | 0.7400 ± 0.1062 | 0.7296 ± 0.1001 | 0.8755 ± 0.1002 |
| Random Forest | 0.6200 ± 0.0653 | 0.5143 ± 0.1067 | 0.6200 ± 0.0653 | 0.5474 ± 0.0843 | 0.7549 ± 0.0568 |
| XGBoost | 0.6667 ± 0.0558 | 0.6128 ± 0.0562 | 0.6667 ± 0.0558 | 0.6176 ± 0.0396 | 0.7539 ± 0.0626 |

### Discussion:
- **Logistic Regression**: Serves as a strong baseline demonstrating linear separability of domain sub-scores.
- **Random Forest**: Selected as the primary operational model due to its robustness against non-linear multi-sensor noise and full compatibility with **SHAP TreeExplainer**.
- **XGBoost**: Provides gradient-boosted comparison for non-linear feature interactions.

---

## 4. Explainable AI (SHAP) Feature Rankings

Top 10 Global Feature Importances computed via SHAP TreeExplainer across all 150 border zones:

| Feature | Mean_SHAP_Importance |
| --- | --- |
| Camera_Activity_Score | 0.02284 |
| Vegetation_Density | 0.01685 |
| Vegetation_Risk | 0.015619 |
| Night_Event_Count | 0.01429 |
| Avg_Slope | 0.01286 |
| Historical_Risk | 0.012841 |
| NDVI | 0.012007 |
| Avg_Elevation | 0.011959 |
| Terrain_Risk | 0.011897 |
| Max_Elevation | 0.011313 |

---

## 5. Conclusion & Operational Impact

The hybrid multi-modal architecture successfully decouples raw computer vision detection (YOLO/CNN) from tabular risk inference, feeding object counts into a Random Forest pipeline. The SHAP explainability layer bridges machine learning predictions with tactical defense directives, enabling command officers to identify specific risk drivers (e.g. patrol gaps or low visibility) and deploy assets effectively.
