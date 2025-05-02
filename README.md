![](UTA-DataScience-Logo.png)

# PCOS Detection using Binary ML Models

This project uses machine learning models to predict whether a patient has PCOS or not using clinical features from a Kaggle tabular dataset.  
[Kaggle Dataset Link](https://www.kaggle.com/datasets/samikshadalvi/pcos-diagnosis-dataset)

---

## Overview
The goal of this project is to predict whether a patient has Polycystic Ovary Syndrome (PCOS) using clinical features and supervised machine learning. The dataset, sourced from Kaggle, includes 1,000 patient records with numerical features such as Age, BMI, Testosterone Level, Antral Follicle Count, and a binary feature for Menstrual Irregularity. The target variable is PCOS_Diagnosis (1 = PCOS, 0 = No PCOS), making this a binary classification task. This project investigates how well these features predict PCOS and how interpretable, reliable models can be built for healthcare applications. While features like Menstrual Irregularity and Antral Follicle Count showed strong correlations with PCOS, others such as BMI were less informative, prompting further experiments without them.

Two models were developed and tested:
* Logistic Regression — selected for its simplicity and interpretability
* Random Forest — chosen for its ability to model non-linear relationships
  
The dataset was split 80/20 into training and testing sets. Both models were evaluated using accuracy, precision, recall, F1 score, ROC-AUC, and confusion matrices. Additionally, 5-fold cross-validation was performed to verify generalizability. Logistic Regression achieved ~91% accuracy, correctly identifying most PCOS cases — a key goal in clinical diagnosis. The Random Forest classifier achieved 100% accuracy on the test set, which raised concerns about overfitting. To investigate this, a smaller Random Forest model with just 3 trees and a depth of 3 was trained. It still achieved ~99.5% accuracy, confirming that strong signals exist in the data and the classes are well-separated. Further steps included visualizing feature correlations, interpreting decision trees, generating confusion matrices, and creating bar and histogram plots to demonstrate class separation. This project shows that with proper preprocessing and analysis, a small clinical dataset can be effectively modeled for disease prediction using simple, interpretable machine learning workflows.
    
---

## Summary of Workdone

### Data
* **Type:** CSV file containing patient health data; output is a binary class: PCOS = 1 or 0.
* **Size:** 1000 rows × ~6 columns (before and after cleaning
* **Split:** 80% train, 20% test using stratified sampling to preserve class balance.

#### Preprocessing / Clean up
* Removed whitespace from column names.
* There were no duplicate or unbalances rows 
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

### Overview of Files in Repository
This repository is structured into two main folders: Final Notebooks and Scratch Notebooks.

Final Notebooks: This folder contains the polished version of the project notebook and relevant outputs:
Total files: 4
* Notebook.ipynb – Main Jupyter notebook containing full implementation, from data preprocessing to model training, evaluation, and interpretation. This is the primary file to reproduce results.
* df_head.txt – Output of df.head() used to show the first few rows of the dataset.
* df_describe.txt – Output of df.describe() summarizing statistical metrics (mean, std, min, max, etc.).
* df_info.txt – Output of df.info() showing column data types and non-null counts.

Scratch notebooks: This folder holds preliminary or experimental work
Total files: 3
* scratch-notebook1.ipynb – Early-stage exploration and initial visualization of the dataset.
* scratch-notebook2.ipynb – Intermediate attempts at cleaning, feature selection, or comparing different model results.
* pcos_dataset.csv – The original dataset used for this project, sourced from Kaggle.

Note: The scratch notebooks folder contains unpolished code meant for brainstorming and iterative development. For the complete and finalized implementation, refer to the Notebook.ipynb in the Final Notebooks folder.

---

### How to Reproduce Results & Software Setup

How to Reproduce Results: 
To reproduce all results in this project, run the Jupyter notebook Notebook.ipynb located in the Final Notebooks folder. This notebook walks through the entire pipeline — from loading the dataset to preprocessing, visualization, model training, evaluation, and cross-validation.

No pre-trained models or cached results are used; everything runs from scratch and is fully documented with markdown explanations throughout the notebook.

Setup Instructions:
* Clone the repository or download the ZIP.
* Open and run Notebook.ipynb in Jupyter Notebook or Visual Studio Code.
* Make sure the dataset pcos_dataset.csv (from the scratch notebooks folder) is present and in the same directory or properly loaded.

Required Python libraries:
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* Can be installed with:
  ```bash
  pip install pandas numpy matplotlib scikit-learn

Notes on Additional Files:The text files df_head.txt, df_info.txt, and df_describe.txt are included to help quickly inspect the dataset structure without needing to load the notebook.
