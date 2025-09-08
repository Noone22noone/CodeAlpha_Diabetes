# 🩺 Diabetes Prediction Model

This project is part of my **CodeAlpha Machine Learning Internship**, where I developed a machine learning model to predict whether a person has diabetes using the **PIMA Indians Diabetes Dataset**.

---

## 📊 Project Highlights
- Predicts likelihood of diabetes based on health and demographic features.
- Implemented and compared **Logistic Regression, SVM, Random Forest, and XGBoost**.
- Evaluated models with **Accuracy, Precision, Recall, F1-Score**.
- Visualized distributions, correlations, and outliers to understand data better.

---

## 🗂 Dataset
- **Source:** [PIMA Indians Diabetes Dataset (Kaggle)](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
- **Features:** Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age  
- **Target Variable:** `Outcome` (1 = Diabetes, 0 = No Diabetes)

---

## 🧠 Methodology
1. **Data Preparation**
   - Loaded dataset with Pandas.
   - Replaced invalid zero values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` with median values.
   - Split data into **80% training** and **20% testing**.

2. **Exploratory Data Analysis**
   - Statistical summary (`.describe()`).
   - Correlation heatmap to identify relationships between features.
   - Boxplots for each feature to detect outliers.

3. **Model Training**
   - **Logistic Regression** (baseline linear model).
   - **Support Vector Machine (SVM)** with linear kernel.
   - **Random Forest Classifier** (bagging ensemble).
   - **XGBoost Classifier** (boosting ensemble).

4. **Evaluation Metrics**
   - Confusion Matrix
   - Classification Report (Precision, Recall, F1-Score)
   - Accuracy comparison table across models

---

## 🛠 Tools & Libraries
- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- Matplotlib, Seaborn  
- Jupyter Notebook  

---

## 📈 Results
| Classifier           | Accuracy | Precision | Recall | F1-score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression  | ~0.77    | ~0.73     | ~0.68  | ~0.70    |
| SVM (Linear)         | ~0.76    | ~0.71     | ~0.66  | ~0.68    |
| Random Forest        | ~0.79    | ~0.74     | ~0.71  | ~0.72    |
| XGBoost              | ~0.82    | ~0.77     | ~0.74  | ~0.75    |

👉 **XGBoost achieved the best overall performance.**

---

## 📷 Visual Outputs
- Correlation heatmap  
- Boxplots of each feature  
- Confusion matrices for each model  

---

## 💼 Internship Details
This project was completed as part of the **CodeAlpha Machine Learning Internship**.
