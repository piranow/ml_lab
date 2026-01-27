# Experiment 2: Binary Classification using Naïve Bayes and K-Nearest Neighbors (KNN)

##  Aim & Objective
To implement Naïve Bayes and K-Nearest Neighbors (KNN) classifiers for a binary classification problem, evaluate them using multiple performance metrics, visualize model behavior, and analyze overfitting, underfitting, and bias–variance characteristics.

---

##  Dataset
**Dataset Name:** Spambase Dataset  
**Task:** Binary Classification (Spam vs Non-Spam)  
**Target Label:**  
- `0` → Not Spam  
- `1` → Spam  

Each row represents one email described using frequency-based numerical features.

Dataset Reference:
- Kaggle: Spambase Dataset

---

##  Preprocessing Steps
1. Loaded dataset into a Pandas DataFrame.
2. Split dataset into:
   - `X` → Feature matrix (all numerical attributes)
   - `y` → Target vector (spam / non-spam labels)
3. Applied **StandardScaler** for feature scaling (important for distance-based algorithms like KNN).
4. Split data into training and testing sets using **train_test_split**.

---

##  Algorithms Implemented
###  Naïve Bayes Classifiers
- Gaussian Naïve Bayes
- Multinomial Naïve Bayes
- Bernoulli Naïve Bayes

###  K-Nearest Neighbors (KNN)
- Baseline KNN
- KNN with KDTree
- KNN with BallTree

###  Hyperparameter Tuning (KNN)
- GridSearchCV
- RandomizedSearchCV

---

##  Visualizations Generated (16 Slots)
The following plots were generated as part of the experiment:

1. Class Distribution (Spam vs Non-Spam)
2. Feature Distribution (Selected Attributes)
3. Confusion Matrix - Gaussian NB
4. ROC Curve - Gaussian NB
5. Confusion Matrix - Multinomial NB
6. ROC Curve - Multinomial NB
7. Confusion Matrix - Bernoulli NB
8. ROC Curve - Bernoulli NB
9. Confusion Matrix - Baseline KNN
10. ROC Curve - Baseline KNN
11. Accuracy vs k (KNN)
12. Training vs Validation Accuracy (GridSearchCV)
13. Confusion Matrix - KDTree KNN
14. ROC Curve - KDTree KNN
15. Confusion Matrix - BallTree KNN
16. ROC Curve - BallTree KNN

---

##  Performance Metrics Used
For each model, the following metrics were calculated:
- Accuracy
- Precision
- Recall
- F1 Score
- Specificity
- False Positive Rate (FPR)
- Training Time (seconds)
- Prediction Time (seconds)
- Confusion Matrix

---

##  Performance Comparison Tables
The following tables were created:

### Table 1: Naïve Bayes Performance Metrics
Compares Gaussian NB, Multinomial NB, and Bernoulli NB using all evaluation metrics.

### Table 2: KNN Hyperparameter Tuning
Shows best `k` and best cross-validation accuracy for:
- Grid Search
- Randomized Search

### Table 3: KNN Performance using BallTree
Performance metrics and time analysis for BallTree-based KNN.

### Table 4: KNN Performance using KDTree
Performance metrics and time analysis for KDTree-based KNN.

### Table 5: Comparison of Neighbor Search Algorithms
Compares KDTree and BallTree based on:
- Accuracy
- Training Time
- Prediction Time
- Memory Usage

---

##  Overfitting and Underfitting Analysis
- **Small k values**:
  - Very high training accuracy
  - Lower validation accuracy
  - High variance → **Overfitting**
- **Moderate k values (optimal k)**:
  - Balanced training and validation accuracy
  - Best generalization performance
  - Good bias–variance trade-off
- **Large k values**:
  - Low training and validation accuracy
  - High bias → **Underfitting**

Hyperparameter tuning using GridSearchCV and RandomizedSearchCV helped select optimal k values and improved generalization.

---

##  Bias–Variance Analysis
### Naïve Bayes
- Assumes conditional independence between features
- High bias, low variance
- Stable performance but may underfit if features are dependent

### KNN
- Model flexibility depends on k
- Small k → low bias, high variance (overfitting)
- Large k → high bias, low variance (underfitting)
- Tuned k gives balanced performance

---

##  Observation & Conclusion
In this experiment, multiple binary classification models were implemented and evaluated using appropriate metrics and visualizations.

- Naïve Bayes classifiers showed stable performance due to low variance but may underfit because of the conditional independence assumption.
- KNN achieved better accuracy when k was properly tuned.
- Small k values caused overfitting, while very large k values caused underfitting.
- Grid Search and Randomized Search successfully identified optimal k values and improved generalization.

Overall, this experiment highlights the importance of model selection and hyperparameter tuning to achieve robust performance and a good bias–variance trade-off.

---

##  Requirements
Install the following libraries before running:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
