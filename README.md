# Corrosion_Prediction_In_Pipelines_ML
# 🔧 Corrosion Prediction in Pipelines using Machine Learning

## 📌 Project Overview

This project presents a **Machine Learning–based corrosion prediction system for pipelines**.
The system predicts corrosion-related metrics based on pipe characteristics, operating conditions, and environmental factors.

The dashboard provides:

* Prediction of **Thickness Loss (mm)**
* Prediction of **Material Loss (%)**
* Prediction of **Corrosion Impact (%)**
* Classification of **Pipe Condition**
* **Remaining Useful Life (RUL)** estimation
* **Explainable AI (SHAP)** for model interpretation
* **Correlation analysis** of features
* **Maintenance recommendations**

The application is implemented using an **interactive Streamlit dashboard**, allowing users to input pipeline parameters and obtain predictions instantly.

---

# 🎯 Objectives

The main objectives of this project are:

* Predict corrosion behavior in pipelines using machine learning
* Estimate pipeline **remaining operational life**
* Provide **interpretable predictions** using SHAP
* Help engineers make **data-driven maintenance decisions**
* Build a **user-friendly dashboard** for corrosion analysis

---

# 🧠 Machine Learning Models

The system uses trained ML models to predict:

| Model                | Output               |
| -------------------- | -------------------- |
| Regression Model     | Thickness Loss (mm)  |
| Regression Model     | Material Loss (%)    |
| Regression Model     | Corrosion Impact (%) |
| Classification Model | Pipe Condition       |

These models are trained using pipeline operational data such as:

* Pipe size
* Wall thickness
* Pressure
* Temperature
* pH
* Flowrate
* Exposure time
* Material type
* Material grade

---

# ⚙️ Features of the Dashboard

## 1️⃣ Corrosion Prediction

The system predicts:

* **Thickness Loss (mm)** — amount of wall thickness lost due to corrosion
* **Material Loss (%)** — percentage of pipe material lost
* **Corrosion Impact (%)** — severity of corrosion damage
* **Pipe Condition** — overall health classification

---

## 2️⃣ SHAP Explainability (Explainable AI)

The dashboard uses **SHAP (SHapley Additive exPlanations)** to explain model predictions.

SHAP shows:

* Which features increase corrosion risk
* Which features reduce corrosion risk
* Feature importance for each prediction

This makes the model **transparent and interpretable**.

---

## 3️⃣ Remaining Useful Life (RUL)

The system estimates **Remaining Useful Life (RUL)** of a pipeline.

RUL is calculated by:

1. Simulating corrosion progression over time
2. Predicting future thickness loss
3. Identifying when thickness loss reaches a **critical threshold**

Critical threshold used:

```
30% wall thickness loss
```

The dashboard displays:

* Remaining useful life in years
* Predicted failure year
* Corrosion progression chart

---

## 4️⃣ Correlation Analysis

The dashboard also provides a **correlation heatmap** using the training dataset.

This analysis helps identify relationships between variables such as:

* Temperature vs corrosion
* pH vs corrosion
* Flowrate vs corrosion

The heatmap helps users understand **which factors most influence corrosion**.

---

## 5️⃣ Maintenance Recommendations

Based on model predictions, the system provides **actionable recommendations**.

Examples:

| Prediction         | Recommendation             |
| ------------------ | -------------------------- |
| Low corrosion      | Routine inspection         |
| Moderate corrosion | Schedule inspection        |
| Severe corrosion   | Immediate maintenance      |
| Critical condition | Pipe repair or replacement |

This makes the system useful for **predictive maintenance planning**.

---

# 🖥️ Dashboard Interface

The Streamlit dashboard allows users to input parameters such as:

* Pipe Size
* Wall Thickness
* Maximum Pressure
* Temperature
* Exposure Time
* pH Level
* Flowrate
* Material Type
* Pipe Grade

After clicking **Predict**, the dashboard displays:

* Corrosion predictions
* Pipe condition
* SHAP explanations
* Remaining useful life
* Correlation analysis
* Maintenance recommendations

---

# 📂 Project Structure

```
Corrosion_Prediction_In_Pipelines_ML
│
├── app.py
├── README.md
├── requirements.txt
│
├── saved_models
│   ├── best_model_Thickness_Loss_mm.pkl
│   ├── best_model_Material_Loss_Percent.pkl
│   ├── best_model_Corrosion_Impact_Percent.pkl
│   ├── best_model_condition_encoded.pkl
│   └── condition_label_encoder.pkl
│
├── data
│   └── market_pipe_thickness_loss_calibrated_ph_flow.csv
│
└── notebook
    └── model_training.ipynb
```

---

# 🛠️ Technologies Used

* Python
* Scikit-learn
* Streamlit
* Pandas
* NumPy
* Matplotlib
* Seaborn
* SHAP
* Joblib

---

# 🚀 Installation and Setup

### 1️⃣ Clone the repository

```
git clone https://github.com/yourusername/Corrosion_Prediction_In_Pipelines_ML.git
```

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit application

```
streamlit run app.py
```

The dashboard will open in your browser.

---

# 📊 Example Use Case

An engineer can input:

* Pipe material
* Operating pressure
* Temperature
* Flowrate
* Exposure time

The system will estimate:

* Expected corrosion damage
* Remaining life of the pipe
* Maintenance recommendations

This enables **predictive maintenance and risk assessment**.

---

# 🔮 Future Improvements

Possible improvements include:

* Real-time sensor data integration
* Deep learning models for corrosion prediction
* Integration with industrial monitoring systems
* Cloud deployment
* Automated inspection scheduling

---

# 👨‍💻 Author

Harsh Jaiswal
Machine Learning & Data Science Enthusiast

---

# 📜 License

This project is developed for **educational and research purposes**.

