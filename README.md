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
Here are the performance scores of each model:

| Classifier           | Accuracy | Precision | Recall  | F1-score |
|----------------------|----------|-----------|---------|----------|
| **XGBoost**          | 0.7338   | 0.6275    | 0.5926  | 0.6095   |
| **Random Forest**    | 0.7403   | 0.6667    | 0.5185  | 0.5833   |
| **SVM (Linear)**     | 0.7078   | 0.6000    | 0.5000  | 0.5455   |
| **Logistic Regression** | 0.7013 | 0.5870    | 0.5000  | 0.5400   |

👉 **XGBoost achieved the highest F1-score, making it the most balanced model in this experiment.**

---

## 📷 Visual Outputs
- Correlation heatmap  
- Boxplots of each feature  
- Confusion matrices for each model  
- Results comparison table (above)

---

## 💼 Internship Details
This project was completed as part of the **CodeAlpha Machine Learning Internship**.
