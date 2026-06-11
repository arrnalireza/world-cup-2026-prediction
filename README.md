# 🏆 World Cup Match Outcome Prediction & Simulation

## 📌 Overview

This project predicts football match outcomes using machine learning and simulates full tournament results using a Monte Carlo approach.

The system combines multiple gradient boosting models and advanced feature engineering to estimate match-level probabilities and tournament progression.

---

## ⚙️ Models Used

- XGBoost Regressor
- LightGBM Regressor
- CatBoost Regressor

An ensemble model is built using optimized weights from out-of-fold predictions.

---

## 🧠 Feature Engineering

Key engineered features include:

- Elo rating difference and Elo-based win probability
- FIFA points and ranking-based metrics
- Team form (last 5 matches)
- Head-to-head historical performance
- Squad value and age differences
- Rest and fatigue indicators

---

## 📊 Validation Strategy

- TimeSeriesSplit cross-validation
- No data shuffling to preserve temporal order
- Out-of-fold evaluation for unbiased performance estimation

---

## 🏆 Results

- Ensemble MSE: *2.896*
- MAE: *1.277*
- R² Score: *0.346*

---

## ⚽ Tournament Simulation

A Monte Carlo simulation is used to generate full tournament outcomes, estimating:

- Stage-by-stage progression probabilities
- Champion likelihood for each team
- Upset and overperformance patterns

---

## 📈 Key Visualizations

- Feature importance comparison (XGB / LGB / CAT)
- Champion probability distribution
- Stage survival curves
- Tournament bracket simulation

---

## Data Sources

This project combines data from multiple football-related sources:

- Historical international match results (https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
- FIFA men's national team rankings
- Transfermarkt squad information and market values
- 2026 FIFA World Cup group and tournament structure

The final dataset was created by cleaning, merging, and feature-engineering information from these sources.

---

## 🚫 Data Availability

Due to licensing restrictions, the dataset is not included in this repository.

To run the project:

must contain the required dataset locally.

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python src/train.py
```

## 👨‍💻 Author
**Alireza Rastegar Nasab(ARRN)**  

📫 GitHub: https://github.com/arrnalireza
📧 Contact: alirezaarrn@gmail.com  