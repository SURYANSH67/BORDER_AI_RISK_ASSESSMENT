# EMBARIA: Border Security Risk Intelligence & Explainable AI Platform

A Research-Grade Hybrid Multi-Modal Machine Learning and Decision-Support Framework for Mountainous Border Defense Operations.

---

## 1. Executive Overview

Border security operations across rugged mountainous terrain require the synthesis of diverse intelligence streams. The **EMBARIA** platform implements a decoupled, hybrid multi-modal architecture that ingests 7 distinct intelligence layers, performs zone-level feature fusion across 150 border sectors (Srinagar Sector: `SGR_ZONE_001` to `SGR_ZONE_150`), evaluates comparative machine learning classifiers under 5-Fold Stratified Cross-Validation, and provides decision-grade explainability via SHAP (Shapley Additive Explanations).

---

## 2. System Architecture

```
                 DATA SOURCES & SIMULATION
-------------------------------------------------------------
Terrain DEM (519k points)  --> terrain_processor.py   ---+
Weather & Visibility       --> weather_processor.py   ---+-> Standardized
Vegetation Remote Sensing  --> vegetation_processor.py---+   150 Border Zones
Patrol Coverage Logs       --> patrol_processor.py    ---+   (SGR_ZONE_001..150)
Historical Infiltration    --> historical_processor.py---+
Camera Events (Simulated)  --> camera_sim.py          ---+
Multi-Sensors (Simulated)  --> sensor_sim.py          ---+
                                                         |
                                                         v
                                              build_master_dataset.py
                                                         |
                                                         v
                                              master_risk_dataset.csv
                                                (150 zones, 48 cols)
                                                         |
                                +------------------------+------------------------+
                                | ML Comparative Engine (5-Fold Stratified CV)    |
                                | - Logistic Regression (Baseline Classifier)     |
                                | - Random Forest (Primary Non-Linear Ensemble)   |
                                | - XGBoost (Gradient Boosting Classifier)        |
                                +------------------------+------------------------+
                                                         |
                                +------------------------+------------------------+
                                | Decision Support & Explainable AI (XAI)         |
                                | - SHAP TreeExplainer Local/Global Attribution   |
                                | - Multi-Modal Ablation Empirical Study          |
                                | - Single-Source Zone Analysis REST API          |
                                | - Full-Stack Interactive GIS Web Application    |
                                +-------------------------------------------------+
```

---

## 3. Key Modalities & Data Processing Pipeline

1. **Terrain Geography (`terrain_processor.py`)**:
   - Ingests `terrain_final.csv` (519,841 coordinate elevation points).
   - Fast spatial binning into 150 grid zones (`Avg_Elevation`, `Avg_Slope`, `Max_Slope`, `Terrain_Risk`).
2. **Weather & Atmospheric Visibility (`weather_processor.py`)**:
   - Consolidates temperature, humidity, pressure, visibility (m), wind speed, clouds, and visibility risk.
3. **Remote-Sensing Vegetation (`vegetation_processor.py`)**:
   - Extracts NDVI (-0.05 to 0.85), vegetation density, and categorical land cover to model concealment risk corridors.
4. **Operational Patrol Shifts (`patrol_processor.py`)**:
   - Aggregates weekly patrol logs: coverage percentage, blind spots, max gap hours, night shifts, and unit readiness score.
5. **Historical Infiltration Records (`historical_processor.py`)**:
   - Processes 5,231 event logs into zone vulnerability indices, night infiltration frequency, and average response times.
6. **Simulated Surveillance Intelligence (`camera_sim.py` & `sensor_sim.py`)**:
   - 1,200 simulated optical/thermal camera detection events.
   - 1,500 simulated multi-sensor telemetry readings (motion, infrared, acoustic, seismic, vibration).
   - *Note: Clearly labeled as simulated intelligence for research transparency.*

---

## 4. Empirical Evaluation & Research Results

### A. 5-Fold Stratified Cross-Validation Benchmark

| Model | Accuracy (Mean +/- Std) | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | ROC-AUC (OVR) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | **0.7400 +/- 0.1062** | **0.7294 +/- 0.1003** | **0.7400 +/- 0.1062** | **0.7296 +/- 0.1001** | **0.8755 +/- 0.1002** |
| **Random Forest (Primary Model)** | 0.6200 +/- 0.0653 | 0.5143 +/- 0.1067 | 0.6200 +/- 0.0653 | 0.5474 +/- 0.0843 | 0.7549 +/- 0.0568 |
| **XGBoost (Comparison Model)** | 0.6667 +/- 0.0558 | 0.6128 +/- 0.0562 | 0.6667 +/- 0.0558 | 0.6176 +/- 0.0396 | 0.7539 +/- 0.0626 |

*Finding: Logistic Regression provides strong linear separation on domain sub-scores, while Random Forest and XGBoost enable non-linear sensor interaction modeling and SHAP TreeExplainer feature attributions.*

---

### B. Multi-Modal Feature Ablation Study

| Experiment Stage | Features | LR F1-Score | LR ROC-AUC | RF F1-Score | RF ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Exp 1: Environmental Only** | 17 | 0.4720 +/- 0.1004 | 0.5808 +/- 0.0963 | 0.5015 +/- 0.0552 | 0.5872 +/- 0.0402 |
| **Exp 2: Env + Operational** | 26 | 0.4967 +/- 0.0519 | 0.6076 +/- 0.0768 | 0.5194 +/- 0.0485 | 0.6732 +/- 0.0460 |
| **Exp 3: Env + Op + Historical** | 32 | 0.5923 +/- 0.0969 | 0.7097 +/- 0.0479 | 0.5361 +/- 0.0700 | 0.6931 +/- 0.0469 |
| **Exp 4: Full Multi-Modal System** | **42** | **0.7296 +/- 0.1001** | **0.8755 +/- 0.1002** | **0.5474 +/- 0.0843** | **0.7549 +/- 0.0568** |

*Empirical Gain: Multi-modal fusion yields a +25.76% boost in F1-score and +29.47% boost in ROC-AUC over environmental features alone.*

---

## 5. Repository Directory Structure

```text
border_infli_ai/
├── app.py                             # Flask Web Server & REST API backend
├── run_pipeline.py                    # Master End-to-End Pipeline Orchestrator
├── demo_walkthrough.py                # 10-Step Interactive Evaluator Demo CLI
├── test_pipeline.py                   # 7/7 Scenario Integration Test Suite
├── run_dashboard.py                   # Standalone GIS Dashboard Server
├── templates/
│   └── index.html                     # Full-Stack Web Application Frontend
├── src/
│   ├── api/
│   │   └── zone_analysis_api.py       # Single-Source Zone Decision Intelligence API
│   ├── preprocessing/
│   │   ├── terrain_processor.py       # Fast GIS terrain grid aggregator (519k points)
│   │   ├── weather_processor.py       # Meteorological feature standardizer
│   │   ├── vegetation_processor.py    # Remote-sensing NDVI & concealment extractor
│   │   ├── patrol_processor.py        # Operational patrol shift aggregator
│   │   └── historical_processor.py    # Infiltration event log processor
│   ├── simulation/
│   │   ├── camera_sim.py              # 1,200 simulated camera triggers generator
│   │   └── sensor_sim.py              # 1,500 simulated multi-sensor readings generator
│   ├── fusion/
│   │   └── build_master_dataset.py    # 48-column master dataset fusion & calibration
│   ├── models/
│   │   ├── train_evaluate.py          # 5-Fold Stratified Cross-Validation engine
│   │   ├── ablation_study.py          # 4-Stage feature ablation experiment runner
│   │   ├── explainability_shap.py     # Global & local SHAP TreeExplainer generator
│   │   └── predict_zone_risk.py       # Real-time single-zone inference engine
│   ├── vision/
│   │   └── yolo_detector.py           # Modular computer vision feature extractor
│   ├── dashboard/
│   │   └── export_dashboard_data.py   # Dashboard JSON payload exporter
│   └── reporting/
│       └── generate_research_report.py# Automated thesis paper compiler
├── data/
│   └── processed/                     # Fused tables and standardized CSVs
├── models/                            # Trained joblib models, metrics & confusion matrices
├── reports/
│   ├── FINAL_THESIS_REPORT.md         # Full 20-Section Academic Thesis Paper
│   └── PROJECT_PRESENTATION_SLIDES.md # 15-Slide Viva / Defense Presentation Deck
└── README.md                          # Repository Documentation
```

---

## 6. Installation & Quickstart Guide

### Prerequisites
- Python 3.10+ (Tested on macOS / Linux / Windows)
- Virtual environment (`venv` recommended)

### Step 1: Clone Repository & Setup Environment
```bash
git clone <repository_url>
cd border_infli_ai
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Execute Master Multi-Modal Pipeline
To re-run the full data preprocessing, simulation, master fusion, 5-fold cross-validation, ablation study, and SHAP explainability:
```bash
python run_pipeline.py
```

### Step 3: Run Interactive Evaluator Demo CLI
To execute the step-by-step evaluation walkthrough:
```bash
python demo_walkthrough.py
```

### Step 4: Run Scenario Integration Tests
```bash
python test_pipeline.py
```

### Step 5: Launch the Interactive Web Application
```bash
python app.py
```
Open your web browser and navigate to: **`http://localhost:7860`**

---

## 7. REST API Endpoints

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/` | `GET` | Serves the interactive GIS command web interface |
| `/api/overview` | `GET` | Returns 150-zone KPI summary counts (Critical, High, Medium, Low) |
| `/api/zones` | `GET` | Returns spatial coordinates and risk levels for all 150 zones |
| `/api/zone/<zone_id>` | `GET` | Returns full multi-modal analysis payload and SHAP drivers |
| `/api/simulate` | `POST` | Ingests live telemetry overrides and returns real-time risk prediction |
| `/api/models` | `GET` | Returns 5-Fold Stratified Cross-Validation evaluation metrics |
| `/api/ablation` | `GET` | Returns multi-modal feature ablation study progression results |
| `/api/shap/global` | `GET` | Returns global top-15 feature importance rankings |
| `/api/datasets/summary`| `GET` | Returns metadata summary for all 7 input datasets |

---

## 8. Operational Scenario Integration Test Suite

Validated across 7 operational test scenarios (`test_pipeline.py`):
1. **Baseline Zone Query**: Standard retrieval and score calculation.
2. **Unseen / Unknown Zone Query Fallback**: Non-existent zone query error handling.
3. **Nighttime Camera Threat Trigger**: Optical/thermal human movement trigger.
4. **Seismic & Acoustic Telemetry Spike**: Ground vibration threshold trigger.
5. **High Patrol Gap Hotspot Scenario**: Extended 23-hour patrol gap vulnerability.
6. **Low Visibility Fog Event**: Adverse weather visibility uncertainty.
7. **Multi-Threat Critical Combination Event**: Multi-domain composite surge.

*Result: 7/7 Scenario Tests Passed (100% Success Rate).*

---

## 9. Academic Thesis & Defense Documentation

- **Full 20-Section Thesis Paper**: [`reports/FINAL_THESIS_REPORT.md`](file:///Users/suryanshdixit/Downloads/border_infli_ai/reports/FINAL_THESIS_REPORT.md)
- **15-Slide Viva / Defense Slide Deck**: [`reports/PROJECT_PRESENTATION_SLIDES.md`](file:///Users/suryanshdixit/Downloads/border_infli_ai/reports/PROJECT_PRESENTATION_SLIDES.md)

---

## 10. Research Disclaimer

This project is an academic research prototype. Environmental, terrain, and patrol data are standardized from public datasets and sector logbooks. Optical camera triggers and sensor telemetry streams are synthetically generated for research validation purposes. The system provides decision-support analytics and is not an autonomous tactical command authority.
