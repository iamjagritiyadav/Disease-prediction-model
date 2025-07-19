# 🧬 Disease Severity Prediction

**A Machine Learning Approach to Predict Cancer Severity Using Regression Models**

This project presents a machine learning pipeline to predict the **severity level of disease** in cancer patients based on clinical and demographic features. The model aims to provide a predictive aid in understanding patient condition levels using structured medical data.

---

## 📁 Dataset Overview

* **File Name**: `cancer patient data sets.csv`
* **Target Feature**: `Level` (severity level of cancer condition)
* **Input Features**: Multiple clinical indicators and patient attributes (e.g., age, gender, medical test results — exact list based on dataset)
* **Prediction Type**: Regression
* **Data Shape**: Parsed using `pandas`

---

## 🔄 Data Preprocessing Pipeline

1. **Data Inspection**

   * Top rows viewed using `.head()`
   * Data types and null values examined via `.info()` and `.isnull().sum()`
   * Statistical summary generated with `.describe()`

2. **Feature and Label Separation**

   * `X`: All features except `Level`
   * `y`: The target variable `Level`

3. **Train-Test Split**

   * Used `train_test_split` with an 80:20 ratio
   * `random_state=100` for reproducibility

---

## 🧠 Model Training and Evaluation

Two regression models were trained and evaluated:

---

### 🔹 Linear Regression

* **Library**: `sklearn.linear_model.LinearRegression`
* **Training**: `.fit(X_train, y_train)`
* **Evaluation Metrics**:

  * Training MSE: `0.046248`
  * Training R² Score: `0.928517`
  * Test MSE: `0.057383`
  * Test R² Score: `0.921597`

📉 *Linear Regression served as a baseline model, providing insight into the linear relationship between features and the severity level.*

---

### 🌲 Random Forest Regressor

* **Library**: `sklearn.ensemble.RandomForestRegressor`
* **Training**: `.fit(X_train, y_train)`
* **Evaluation Metrics**:

  * Training MSE: `0.000001`
  * Training R² Score: `0.999998`
  * Test MSE: `0.000001`
  * Test R² Score: `0.999999`

✅ *The Random Forest model dramatically outperformed the linear model, capturing complex, non-linear feature interactions with near-perfect accuracy.*

---

## 📊 Metrics Summary

| Model             | Training MSE | Training R² | Test MSE | Test R²  |
| ----------------- | ------------ | ----------- | -------- | -------- |
| Linear Regression | 0.046248     | 0.928517    | 0.057383 | 0.921597 |
| Random Forest     | 0.000001     | 0.999998    | 0.000001 | 0.999999 |

---

## 🧪 Tools and Libraries

* **Data Manipulation**: `pandas`, `numpy`
* **Modeling**: `scikit-learn`
* **Environment**: Google Colab

---

## 🔍 Key Insights and Learnings

* Built a full machine learning pipeline from raw data to evaluation
* Demonstrated the performance gap between linear and ensemble models
* Learned how to quantify model quality using MSE and R²
* Gained hands-on experience in using machine learning for healthcare-related prediction tasks

---

## 👩‍💻 Author

**Jagriti Yadav**
*Aspiring Data Scientist focused on healthcare analytics and machine learning solutions.*
