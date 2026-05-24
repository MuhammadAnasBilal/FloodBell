# 🌊 FloodGuard Pakistan — AI-Powered Disaster Management System

> **BSAI Combined Project: Introduction to AI + Machine Learning**
> Flood Risk Prediction & Intelligent Disaster Response for Pakistan

---

## 🎯 Problem Statement

Given **20 environmental/infrastructure risk factors** for a region in Pakistan, predict the **probability of flooding (0–1)** and generate **intelligent evacuation plans, resource allocation, and disaster response recommendations** using AI techniques.

**Dataset**: [Kaggle Playground Series S4E5](https://www.kaggle.com/competitions/playground-series-s4e5/data)
- ~1.1M training samples, 20 features, target: `FloodProbability`
- Features represent risk indices (0–10) for monsoon intensity, drainage, deforestation, urbanization, etc.

---

## 🏗️ System Architecture

```
Raw Data → Preprocessing → ML Models → Flood Probability
                                           ↓
                              ┌─────────────┼─────────────┐
                              ↓             ↓             ↓
                         A* Agent      CSP Solver    Knowledge Base
                     (Evacuation)   (Resources)   (Recommendations)
                              ↓             ↓             ↓
                              └─────────────┼─────────────┘
                                           ↓
                                    Web Dashboard
```

---

## 📋 Project Steps (All 10 Completed)

| Step | Description | File(s) |
|------|-------------|---------|
| 1 | Problem Statement | This README |
| 2 | EDA & Preprocessing | `notebooks/01_eda_preprocessing.py` |
| 3 | Three ML Models | `notebooks/02_ml_models.py` |
| 4 | Bias-Variance & Regularization | `notebooks/03_bias_variance.py` |
| 5 | Model Evaluation Metrics | `notebooks/04_evaluation.py` |
| 6 | Unsupervised Learning (Clustering) | `notebooks/05_clustering.py` |
| 7 | Intelligent Agent (A* Search) | `src/agent/flood_agent.py` |
| 8 | CSP (Backtracking Search) | `src/csp/resource_csp.py` |
| 9 | Knowledge-Based System | `src/knowledge_base/flood_kb.py` |
| 10 | System Integration (Dashboard) | `app.py` + `templates/` + `static/` |

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the ML pipeline (in order)
python notebooks/01_eda_preprocessing.py
python notebooks/02_ml_models.py
python notebooks/03_bias_variance.py
python notebooks/04_evaluation.py
python notebooks/05_clustering.py

# 3. Test AI components
python src/agent/flood_agent.py
python src/csp/resource_csp.py
python src/knowledge_base/flood_kb.py

# 4. Launch the dashboard
python app.py
# Open http://127.0.0.1:5000
```

---

## 📁 Project Structure

```
ML_Prj/
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── app.py                              # Flask web application (Step 10)
├── data/
│   ├── train.csv                       # Training data (Kaggle)
│   ├── test.csv                        # Test data (Kaggle)
│   └── processed/                      # Preprocessed splits
├── notebooks/
│   ├── 01_eda_preprocessing.py         # Step 2
│   ├── 02_ml_models.py                 # Step 3
│   ├── 03_bias_variance.py             # Step 4
│   ├── 04_evaluation.py                # Step 5
│   └── 05_clustering.py                # Step 6
├── src/
│   ├── agent/flood_agent.py            # Step 7: A* Search Agent
│   ├── csp/resource_csp.py             # Step 8: CSP Solver
│   └── knowledge_base/flood_kb.py      # Step 9: Forward Chaining KB
├── models/                             # Saved trained models (.pkl)
├── outputs/                            # Generated plots and metrics
├── templates/index.html                # Dashboard HTML
└── static/
    ├── css/style.css                   # Dashboard styles
    └── js/app.js                       # Dashboard interactivity
```

---

## 🤖 ML Models

| Model | Type | R² Score | Description |
|-------|------|----------|-------------|
| Ridge Regression | Baseline | ~0.78 | Linear model with L2 regularization |
| Random Forest | Intermediate | ~0.85 | Ensemble of decision trees |
| XGBoost | Advanced | ~0.87 | Gradient boosted trees (state-of-art) |

---

## 🧠 AI Components

### Step 7: Intelligent Agent (A* Search)
- **PEAS**: Performance (minimize casualties), Environment (Pakistan flood zones), Actuators (alerts/shelters/teams), Sensors (ML predictions)
- **State Space**: Initial state from ML prediction → Goal state (all evacuated)
- **Search**: A* with admissible heuristic based on unmet needs
- **Output**: Optimal sequence of evacuation actions

### Step 8: CSP (Constraint Satisfaction)
- **Variables**: Shelter assignments, team deployments, supply allocation, evacuation priority
- **Constraints**: Capacity limits, coverage requirements, no-conflict rules
- **Solver**: Backtracking with Forward Checking + MRV heuristic

### Step 9: Knowledge-Based System
- **12 IF-THEN rules** covering risk classification, evacuation triggers, dam breach warnings, landslide alerts
- **Forward chaining** inference engine with full trace
- **Two contrasting examples**: High-risk (8+ rules fire) vs Low-risk (1 rule fires)

---

## 🛡️ Technology Stack

- **ML**: scikit-learn, XGBoost, pandas, numpy
- **Visualization**: matplotlib, seaborn, Chart.js
- **Web**: Flask, Leaflet.js (maps), HTML/CSS/JS
- **AI**: Custom implementations (no external AI libraries)
