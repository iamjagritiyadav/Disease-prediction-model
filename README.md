# 🧬 Disease Severity Prediction

**A Machine Learning Approach to Predict Cancer Severity Using Regression Models**

This project presents a machine learning pipeline to predict the **severity level of disease** in cancer patients based on clinical and demographic features. The model aims to provide a predictive aid in understanding patient condition levels using structured medical data.

---

## 📁 Dataset Overview

* **File Name**: `cancer patient data sets.csv`
* **Target Feature**: `Level` (severity level of cancer condition)
* **Input Features**: Multiple clinical indicators and patient attributes (e.g., age, gender, medical test results — exact list based on dataset)
* **Prediction Type**: Regression
* **Data Shape**: Implicit from the dataset; parsed using `pandas`

---

## 🔄 Data Preprocessing Pipeline

1. **Data Inspection**

   * Displayed top rows using `.head()`
   * Reviewed column types and null values using `.info()` and `.isnull().sum()`
   * Summary statistics checked with `.describe()`

2. **Feature and Label Separation**

   * Target variable (`y`): `Level`
   * Features (`X`): All remaining columns after dropping `Level`

3. **Train-Test Split**

   * Utilized `train_test_split` from `sklearn.model_selection`
   * Split ratio: **80% training**, **20% testing**
   * Random seed: `random_state=100` to ensure reproducibility

---

## 🧠 Model Training and Evaluation

Two regression models were implemented and compared for predictive performance:

---

### 🔹 Linear Regression

* **Library**: `sklearn.linear_model.LinearRegression`
* **Training**: `.fit(X_train, y_train)`
* **Prediction**: `.predict(X_test)`
* **Performance Metrics**:

  * **Training MSE**: Computed with `mean_squared_error(y_train, y_pred_train)`
  * **Testing MSE**
  * **Training R² Score**: `r2_score(y_train, y_pred_train)`
  * \*\*Testing R² Score\`

📉 *Linear Regression served as a baseline model and established the fundamental relationship between features and target.*

---

### 🌲 Random Forest Regressor

* **Library**: `sklearn.ensemble.RandomForestRegressor`
* **Training**: `.fit(X_train, y_train)`
* **Prediction**: `.predict(X_test)`
* **Performance Metrics**:

  * **Training MSE & R²**
  * **Testing MSE & R²**

✅ *The Random Forest model significantly improved test performance, capturing non-linear interactions between variables.*

---

## 📊 Metrics Summary

| Model             | Train MSE | Train R² | Test MSE | Test R² |
| ----------------- | --------- | -------- | -------- | ------- |
| Linear Regression | (value)   | (value)  | (value)  | (value) |
| Random Forest     | (value)   | (value)  | (value)  | (value) |

> *(Insert actual values from your model results to complete the table.)*

---

## 🧪 Tools and Libraries

* **Data Manipulation**: `pandas`, `numpy`
* **Modeling**: `scikit-learn` (LinearRegression, RandomForestRegressor, train\_test\_split, metrics)
* **Environment**: Executed on **Google Colab** — ideal for interactive development, scalable cloud execution, and GPU support if extended

---

## 🔍 Key Insights and Learnings

* Implemented a full machine learning workflow: **data exploration → preprocessing → model training → evaluation**
* Understood the limitations of linear models on complex datasets
* Gained experience in tuning and evaluating **regression models**
* Learned to interpret **R²** and **MSE** to compare model accuracy and generalization
* Built the foundation for more advanced ML applications in healthcare prediction tasks

---

## 👩‍💻 Author

**Jagriti Yadav**
*Aspiring Data Scientist with a focus on machine learning applications in healthcare and predictive analytics.*
