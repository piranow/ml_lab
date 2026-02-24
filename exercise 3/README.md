# Loan Sanction Amount Prediction - Machine Learning Project

## 📌 Project Overview

This project focuses on predicting the **Loan Sanction Amount (USD)**
using machine learning regression models. The dataset contains various
applicant and financial attributes that influence the loan approval
amount.

The notebook performs: - Data loading and preprocessing - Handling
missing values - Feature transformation (Encoding & Scaling) - Model
training using multiple regression algorithms - Model evaluation using
standard regression metrics - Hyperparameter tuning using GridSearchCV

------------------------------------------------------------------------

## 📂 Dataset

The project uses: - `train.csv` -- Training dataset - `test.csv` --
Testing dataset

The target variable: - **Loan Sanction Amount (USD)**

------------------------------------------------------------------------

## 🛠 Technologies & Libraries Used

-   Python 3.x
-   NumPy
-   Pandas
-   Matplotlib
-   Seaborn
-   Scikit-learn

------------------------------------------------------------------------

## ⚙️ Project Workflow

### 1️⃣ Data Preprocessing

-   Load training and test datasets
-   Check for missing values
-   Drop rows with missing target values
-   Separate numerical and categorical columns
-   Apply:
    -   `SimpleImputer` for missing values
    -   `StandardScaler` for numerical scaling
    -   `OneHotEncoder` for categorical encoding

### 2️⃣ Model Building

The following regression models are used: - Linear Regression - Ridge
Regression - Lasso Regression - ElasticNet

### 3️⃣ Model Evaluation

Models are evaluated using: - Mean Absolute Error (MAE) - Mean Squared
Error (MSE) - Root Mean Squared Error (RMSE) - R² Score

### 4️⃣ Hyperparameter Tuning

-   GridSearchCV is used to find the best model parameters.
-   Cross-validation is performed for better generalization.

------------------------------------------------------------------------

## 📊 Performance Metrics

| Metric \| Description \|

\|--------\|-------------\| MAE \| Average absolute prediction error \|
\| MSE \| Average squared prediction error \| \| RMSE \| Square root of
MSE \| \| R² Score \| Model goodness of fit \|

------------------------------------------------------------------------

## 🚀 How to Run the Project

1.  Install required libraries:

    ``` bash
    pip install numpy pandas matplotlib seaborn scikit-learn
    ```

2.  Place dataset files inside:

        dataset/train.csv
        dataset/test.csv

3.  Open and run the notebook:

        Ex - 3.ipynb

------------------------------------------------------------------------

## 📈 Expected Outcome

The model predicts the Loan Sanction Amount based on applicant details
and financial information. The best-performing regression model is
selected based on evaluation metrics.

------------------------------------------------------------------------

## 👨‍💻 Author

Developed as part of Machine Learning Lab Exercise.

------------------------------------------------------------------------

## 📌 Notes

-   Ensure dataset path is correct before running.
-   Perform proper data cleaning if new data is used.
-   Hyperparameter tuning improves model performance.

------------------------------------------------------------------------

⭐ If you found this project helpful, feel free to improve and
experiment with additional models like Random Forest or XGBoost.
