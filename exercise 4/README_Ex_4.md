# Spam Email Detection - Machine Learning Project

## 📌 Project Overview

This project focuses on **Spam Email Detection** using machine learning
classification techniques.

The dataset is taken from the Spambase dataset (Kaggle version) and
contains various word frequency and email content features used to
classify emails as:

-   **Ham (0)** -- Not Spam\
-   **Spam (1)** -- Spam Email

The notebook includes: - Data loading and exploration - Data
visualization - Feature scaling - Model training and evaluation -
Performance analysis using classification metrics

------------------------------------------------------------------------

## 📂 Dataset

File Used: - `dataset/spambase_csv_Kaggle.csv`

Target Column: - `class` - 0 → Ham - 1 → Spam

------------------------------------------------------------------------

## 🛠 Technologies & Libraries Used

-   Python 3.x
-   Pandas
-   Matplotlib
-   Seaborn
-   Scikit-learn

------------------------------------------------------------------------

## ⚙️ Project Workflow

### 1️⃣ Data Loading

-   Load dataset using Pandas
-   Display head and shape of dataset
-   Check for missing values
-   Generate descriptive statistics

### 2️⃣ Data Visualization

-   Count plot for class distribution (Spam vs Ham)
-   Boxplots for important features such as:
    -   `word_freq_free`
    -   `word_freq_money`
    -   `word_freq_business`
    -   `capital_run_length_total`

Visualizations are saved as: - `classDistribution.png` -
`boxPlotsImportantFeatures.png`

### 3️⃣ Data Preprocessing

-   Feature Scaling using `StandardScaler`
-   Train-Test split using `train_test_split`

### 4️⃣ Model Evaluation Metrics

The following metrics are used:

-   Accuracy
-   Precision
-   Recall
-   F1 Score
-   Confusion Matrix

Confusion matrix visualization is done using: - `ConfusionMatrixDisplay`

------------------------------------------------------------------------

## 📊 Performance Metrics Explanation

| Metric \| Description \|

\|--------\|-------------\| Accuracy \| Overall correctness of the model
\| \| Precision \| Correctly predicted spam out of total predicted spam
\| \| Recall \| Correctly predicted spam out of actual spam \| \| F1
Score \| Harmonic mean of Precision and Recall \|

------------------------------------------------------------------------

## 🚀 How to Run the Project

1.  Install required libraries:

    ``` bash
    pip install pandas matplotlib seaborn scikit-learn
    ```

2.  Place dataset inside:

        dataset/spambase_csv_Kaggle.csv

3.  Open and run:

        Ex - 4.ipynb

------------------------------------------------------------------------

## 📈 Expected Outcome

The trained model classifies emails as Spam or Ham based on word
frequency and text-based features.\
Performance is evaluated using standard classification metrics and
confusion matrix.

------------------------------------------------------------------------

## 👨‍💻 Author

Developed as part of Machine Learning Laboratory Exercise.

------------------------------------------------------------------------

## 📌 Notes

-   Ensure correct dataset path before running the notebook.
-   Feature scaling is important for better performance.
-   You can extend this project using advanced models like:
    -   Logistic Regression
    -   Random Forest
    -   Support Vector Machine (SVM)
    -   XGBoost

------------------------------------------------------------------------

⭐ Feel free to enhance this project with feature engineering or
hyperparameter tuning.
