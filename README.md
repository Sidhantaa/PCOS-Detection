![](UTA-DataScience-Logo.png)

# PCOS Detection using Binary ML Models

* **One Sentence Summary**  
This project uses machine learning models to predict whether a patient has PCOS or not using clinical features from a Kaggle tabular dataset.  
[Kaggle Dataset Link](https://www.kaggle.com/datasets/samikshadalvi/pcos-diagnosis-dataset)

---

## Overview

* **Task:** Predict whether a patient has PCOS (binary classification) based on clinical and hormonal features.
* **Approach:** Trained logistic regression and random forest models on a clean, labeled dataset of 1000 rows. Applied data visualization, feature analysis, and model evaluation including confusion matrix and cross-validation.
* **Performance:** 
  - Logistic Regression: ~91.5% accuracy  
  - Random Forest: 100% accuracy (with further checks for overfitting using smaller trees and cross-validation)
    
---

## Summary of Workdone

### Data
* **Type:** CSV file containing patient health data; output is a binary class: PCOS = 1 or 0.
* **Size:** 1000 rows × ~13 columns after cleaning.
* **Split:** 80% train, 20% test using stratified sampling to preserve class balance.

#### Preprocessing / Clean up
* Removed whitespace from column names.
* Dropped duplicate or unlabeled rows if any.
* No missing values in dataset.
* Scaled numerical features using StandardScaler.

#### Data Visualization
* Bar plots for binary features like `Menstrual_Irregularity`
* Box plots and histograms for numerical features
* Correlation heatmap to identify strong predictors

---

### Problem Formulation

* **Input:** Numerical and binary features related to hormones and clinical signs.
* **Output:** Binary classification — whether the patient has PCOS.
* **Models Used:**
  - Logistic Regression (baseline)
  - Random Forest (and a simplified version)
* **Hyperparameters:** RandomForest with max_depth=3, n_estimators=3 for interpretability testing

---

### Training

* Used `train_test_split` with stratify on target
* Trained in Jupyter Notebook using scikit-learn on a MacBook Pro
* Training time was negligible due to small dataset size
* Used 5-fold cross-validation for both models

---

### Performance Comparison

| Model              | Accuracy | F1 Score | Precision | Recall |
|-------------------|----------|----------|-----------|--------|
| Logistic Regression | 91.5%   | 0.80     | 0.76      | 0.85   |
| Random Forest       | 100%    | 1.00     | 1.00      | 1.00   |

* ROC curves not generated, but confusion matrices used to visualize class performance.

---

### Conclusions

* Both models performed well; Random Forest gave perfect results but showed signs of potential overfitting.
* Even a very small tree model achieved ~99.5%, indicating strong signal in the data.
* Features like `Menstrual_Irregularity` were near-perfect predictors.

---

### Future Work

* Try external validation or test on a new PCOS dataset.
* Consider clinical deployment — using decision trees to explain predictions to doctors.
* Explore SHAP values or feature importance visualizations for model transparency.

---

## How to Reproduce Results

* Run `project.ipynb` from top to bottom in a Jupyter environment.
* Make sure the dataset file (`pcos_dataset.csv`) is in the same directory.
* All code uses standard libraries from scikit-learn, pandas, matplotlib.

---

### Overview of Files in Repository

* `project.ipynb` – Full notebook with data cleaning, EDA, training, and evaluation.
* `pcos_dataset.csv` – Input dataset used for this classification task.
* `README.md` – Overview of the project, motivation, steps, and findings.

---

### Software Setup

* Python 3.9+
* Required packages:
  - pandas
  - numpy
  - matplotlib
  - scikit-learn
* Can be installed with:
  ```bash
  pip install pandas numpy matplotlib scikit-learn



