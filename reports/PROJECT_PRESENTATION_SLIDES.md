# EMBARIA: Border Security Risk Intelligence & Explainable AI
## Viva & Final Defense Presentation Slides

---

### Slide 1: Title & Project Overview
- **Project**: EMBARIA — Hybrid Multi-Modal Machine Learning Architecture for Border Security Risk Assessment & Decision Support
- **Focus Sector**: Srinagar Sector (150 Spatial Zones: `SGR_ZONE_001` to `SGR_ZONE_150`)
- **Key Paradigms**: Multi-Modal Tabular Fusion + 5-Fold Stratified Cross-Validation + TreeExplainer SHAP + Interactive GIS Dashboard

---

### Slide 2: Problem Statement & Operational Challenges
- **Mountainous Border Challenges**: High-altitude ridges, adverse weather, vegetation concealment corridors, and physical patrol gaps.
- **Current Limitations**: Isolated, siloed data streams; black-box neural networks lacking explainability; misapplication of end-to-end CNNs directly on tabular sensor signals.

---

### Slide 3: Proposed Hybrid Multi-Modal Architecture
- **Architecture Principle**: Decouple Vision/Sensor feature extraction from Tabular Risk Modeling.
- **Workflow**:
  1. Optical/Thermal Camera $\to$ Object Detections (YOLO/CNN) $\to$ Tabular Metadata
  2. Multi-Sensors $\to$ Acoustic / Seismic Activity Scores
  3. Preprocessing $\to$ Terrain (519k points) + Weather + Vegetation + Patrol + Historical
  4. Master Fusion $\to$ 150 Zones $\times$ 42 Features
  5. Machine Learning Comparison $\to$ Logistic Regression, Random Forest, XGBoost
  6. Explainable AI $\to$ SHAP Local & Global Attributions

---

### Slide 4: Data Preprocessing & Modality Harmonization
- **Terrain GIS**: 519,841 raw points binned into a 150-zone spatial grid.
- **Weather & Visibility**: Temperature, Humidity, Pressure, Wind Speed, Visibility (meters).
- **Vegetation / Remote Sensing**: NDVI & Land Cover concealment modeling.
- **Operational Patrol**: Shift coverage %, blind spots, and max gap hours.
- **Historical Infiltration**: 5,231 event logs aggregated into frequency & response times.
- **Simulated Surveillance**: 1,200 camera triggers + 1,500 multi-sensor readings.

---

### Slide 5: Multi-Modal Feature Fusion & Target Calibration
- **Master Dataset**: 150 zones $\times$ 48 columns (42 predictive features).
- **Domain Weighting Formulation**:
  $$Risk = 0.25 \cdot ERS + 0.25 \cdot ORS + 0.25 \cdot HRS + 0.25 \cdot SIS$$
- **Balanced Multi-Class Distribution**:
  - Medium Risk: 86 Zones (57.3%)
  - High Risk: 40 Zones (26.7%)
  - Low Risk: 19 Zones (12.7%)
  - Critical Risk: 5 Zones (3.3%)

---

### Slide 6: Machine Learning Methodology & 5-Fold Cross-Validation
- **Rigorous Evaluation**: 5-Fold Stratified Cross-Validation across all models.
- **Metrics Evaluated**: Accuracy, Weighted Precision, Weighted Recall, Weighted F1-Score, Multi-Class ROC-AUC (OVR).

| Model | Accuracy (Mean ± Std) | F1-Score (Mean ± Std) | ROC-AUC (Mean ± Std) |
| :--- | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | **0.7400 ± 0.1062** | **0.7296 ± 0.1001** | **0.8755 ± 0.1002** |
| **Random Forest (Primary)** | 0.6200 ± 0.0653 | 0.5474 ± 0.0843 | 0.7549 ± 0.0568 |
| **XGBoost (Comparison)** | 0.6667 ± 0.0558 | 0.6176 ± 0.0396 | 0.7539 ± 0.0626 |

---

### Slide 7: Research Finding — Model Comparison Nuance
- **Finding**: Logistic Regression achieves the highest linear separation on the composite domain sub-scores.
- **Role of Random Forest & XGBoost**: Retained as non-linear models capable of capturing non-linear sensor telemetry interactions and providing tree paths for SHAP explanations.

---

### Slide 8: Multi-Modal Feature Ablation Study (Empirical Proof)
- **Research Question**: *Does multi-modal fusion improve infiltration risk prediction over isolated data sources?*

| Experiment Stage | Features | LR F1-Score | LR ROC-AUC |
| :--- | :---: | :---: | :---: |
| **Exp 1: Environmental Only** | 17 | 0.4720 ± 0.1004 | 0.5808 ± 0.0963 |
| **Exp 2: Env + Operational** | 26 | 0.4967 ± 0.0519 | 0.6076 ± 0.0768 |
| **Exp 3: Env + Op + Historical** | 32 | 0.5923 ± 0.0969 | 0.7097 ± 0.0479 |
| **Exp 4: Full Multi-Modal System** | **42** | **0.7296 ± 0.1001** | **0.8755 ± 0.1002** |

- **Empirical Gain**: **+25.76% F1 gain** and **+29.47% ROC-AUC gain** from multi-modal fusion.

---

### Slide 9: Explainable AI (SHAP TreeExplainer)
- **Global Feature Ranking**: Top drivers identified as Historical Risk, Patrol Risk, Terrain Risk, Camera Activity, and Vegetation Density.
- **Local Zone Attribution**: Automatically generates plain-text explanation of risk escalators (+SHAP) vs risk mitigators (-SHAP).

---

### Slide 10: Computer Vision Modular Extractor
- **Architecture**: Camera Frame $\to$ YOLO Object Detection $\to$ Human/Vehicle Counts & Confidence $\to$ Tabular Random Forest Model.
- Demonstrates proper computer vision decoupling without end-to-end black-box risk prediction.

---

### Slide 11: Real-Time Tactical Action API
- Single-zone live inference module ([`src/models/predict_zone_risk.py`](file:///Users/suryanshdixit/Downloads/border_infli_ai/src/models/predict_zone_risk.py)).
- Produces automated command directives:
  - **RED Alert**: Scramble reconnaissance UAV, deploy Quick Reaction Team (QRT).
  - **AMBER Alert**: Increase thermal pan frequency, dispatch foot patrol.
  - **YELLOW Alert**: Routine sensor monitoring.
  - **GREEN Alert**: Standard surveillance posture.

---

### Slide 12: GIS Border Security Intelligence Dashboard
- **Interactive UI**: Leaflet GIS Map with 150 zone markers and color-coded risk levels.
- **9 Specialized Pages**: Overview KPIs, GIS Map, Zone Analysis, SHAP Drivers, Dataset Explorer, Model Analytics, Ablation Study, Surveillance Telemetry, and Zone Comparison Tool.

---

### Slide 13: Operational Scenario Integration Testing
- **Result**: 7/7 Scenario Tests Passed (100% Success Rate):
  1. Baseline Zone Query $\to$ PASSED
  2. Unknown Zone Fallback $\to$ PASSED
  3. Night Camera Trigger $\to$ PASSED
  4. Sensor Spike $\to$ PASSED
  5. Patrol Gap $\to$ PASSED
  6. Low Visibility Fog $\to$ PASSED
  7. Multi-Threat Critical Event $\to$ PASSED

---

### Slide 14: Limitations & Ethical Transparency
- **Research Scope**: Validated on 150 spatial border zones.
- **Synthetic Data**: Camera and sensor streams are explicitly declared as simulated intelligence.
- **Academic Rigor**: Evaluated with cross-validation and transparent ablation studies rather than unsubstantiated claims.

---

### Slide 15: Conclusion & Future Work
- **Summary**: Delivered a fully functional, scientifically validated, explainable decision support prototype.
- **Future Directions**: Satellite Synthetic Aperture Radar (SAR) integration, reinforcement learning patrol route optimization, and edge deployment on NVIDIA Jetson embedded hardware.
