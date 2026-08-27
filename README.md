# 🌙 Sleep Track — For tracking sleep health (NTI Project)


🚀 **Live Demo:** [sleep-track.streamlit.app](https://sleep-track.streamlit.app/)

---

## Overview

Sleep Track is a full machine learning web application built to analyze sleep habits and predict sleep disorder risks based on physiological metrics and lifestyle data (such as BMI, sleep latency, REM/deep sleep ratios, resting heart rate, screen time, and stress levels).

The project includes exploratory data analysis, a benchmark comparison across multiple classification models, an interactive patient assessment tool, and a raw data explorer.

---

## Features

* **Exploratory Data Analysis (EDA):** Interactive Plotly charts showing class distributions, correlation between sleep duration and stress, BMI breakdowns, and bedtime screen habits.
* **Model Benchmark:** Performance comparison across 5 supervised classifiers:
  * Random Forest
  * XGBoost
  * Logistic Regression
  * Decision Tree
  * K-Nearest Neighbors (KNN)
  * Evaluation includes: Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrices.
* **Risk Predictor:** Form interface to input patient biometrics and generate instant risk predictions along with probability gauges.

---

## Tech Stack

* **Language:** Python
* **Web App:** Streamlit
* **Data & Visualization:** Pandas, NumPy, Plotly
* **Machine Learning:** Scikit-Learn, XGBoost

---

## Project Structure

```text
├── .streamlit/
│   └── config.toml                        # Streamlit theme & UI config
├── sleep_health_dataset.csv               # Dataset file
├── Sleep_Health_ML_Project_Final.ipynb    # Data exploration, preprocessing & model experiments
├── app.py                                 # Main Streamlit application
├── requirements.txt                       # Project dependencies
└── README.md                              # Documentation
