![](UTA-DataScience-Logo.png)

# PCOS Detection using Binary ML Models
***This project uses machine learning models to predict whether a patient has PCOS or not using clinical features from a Kaggle tabular dataset.***  

---

## Overview
The goal of this project is to predict whether a patient has Polycystic Ovary Syndrome (PCOS) using clinical features and supervised machine learning. The dataset, sourced from Kaggle, includes 1,000 patient records with numerical features such as Age, BMI, Testosterone Level, Antral Follicle Count, and a binary feature for Menstrual Irregularity. The target variable is PCOS_Diagnosis (1 = PCOS, 0 = No PCOS), making this a binary classification task. This project investigates how well these features predict PCOS and how interpretable, reliable models can be built for healthcare applications. While features like Menstrual Irregularity and Antral Follicle Count showed strong correlations with PCOS, others such as BMI were less informative, prompting further experiments without them.

---

## Summary of Work Done

### Data
* Type: CSV file from Kaggle
* Input: Clinical indicators including Age, BMI, Menstrual Irregularity, Testosterone Level, and Antral Follicle Count
* Output: Binary label (PCOS Diagnosis) — 1 indicates PCOS, 0 indicates no PCOS
* Size: 1000 patient records with 6 columns total

Split:
* 80% for training (800 samples)
* 20% for testing (200 samples)

5-Fold Cross-Validation was applied to assess model consistency

### Preprocessing / Cleanup
* No missing values were found in the dataset, so no imputation was needed
* Removed any unnamed or ID columns that did not contribute to learning
* All features were already numerical (either binary or continuous), so no encoding was necessary
* No duplicated rows were present after manual inspection
* Feature scaling was applied to continuous variables to support model training (Logistic Regression in particular)

### Data Visualization

* Histograms with KDE curves were used to examine the distribution of each numerical feature. This helped us assess the spread, central tendency, and skewness of key variables like Age, BMI, Testosterone Level, Antral Follicle Count, and Menstrual Irregularity.
  
Graph 1.1: Distribution of Numerical Features in the Dataset 
<img width="1113" alt="Screenshot 2025-05-02 at 11 15 18 AM" src="https://github.com/user-attachments/assets/8eb907ee-6e0c-4dc8-9902-b842233fc98b" />


Graph 1.2: Distirbution of Numerication Features by PCOS Class
<img width="1114" alt="Screenshot 2025-05-02 at 11 16 11 AM" src="https://github.com/user-attachments/assets/77d86f84-601b-4061-a0e6-7fb2dbdae4b0" />

* Bar plots grouped by PCOS diagnosis were generated for visualizing how each feature behaves across PCOS-positive and PCOS-negative groups.
* We observed that features like Menstrual Irregularity and Antral Follicle Count show clear separation between classes, making them highly predictive.
* BMI showed overlapping distributions between PCOS and non-PCOS groups, suggesting it may be less discriminative and possibly redundant.
* Visualizations guided later modeling decisions, including trying a model without BMI to test its impact.
* All plots support the conclusion that the dataset contains well-separated classes, making it suitable for classification.

### Problem Formulation
* This project aims to predict PCOS (Polycystic Ovary Syndrome) using five clinical features: Age, BMI, Menstrual Irregularity, Testosterone Level, and Antral Follicle Count. The target is binary: 1 for PCOS, 0 for non-PCOS.
* Since the dataset was already clean and numeric, no imputation or encoding was required — we could directly train classification models.
* We tested two main models:
  * Logistic Regression: A simple, interpretable baseline.
  * Random Forest: A more flexible model that handles non-linear patterns and ranks feature importance.
* To check if model complexity was necessary, we also trained a small Random Forest with just 3 trees and shallow depth. It still performed well, helping rule out overfitting concerns.
* Finally, 5-fold cross-validation was used to test generalization across splits, and we evaluated models using accuracy, precision, recall, F1 score, and confusion matrices.

### Training
* Model training was performed using Python with scikit-learn in a Jupyter Notebook environment on a local machine.
* The dataset is small (1000 rows), so training time was minimal — both models completed training in under a second.
* Key training details:
  * No categorical encoding was needed, as all features were either binary or continuous.
  * No epochs or iterative training loops were required — both models are deterministic and do not require gradient descent.
  * Feature scaling was applied to continuous variables to improve the performance of Logistic Regression.
* There were no significant preprocessing challenges. The high accuracy and clean structure of the dataset made this project an ideal demonstration of how basic machine learning models can be used effectively in binary health 
  classification tasks.
* * Used 5-fold cross-validation for both models

### Performance Comparison

#### Logistic regression
* Accuracy: 91%
* Classification Report
<img width="460" alt="Screenshot 2025-05-02 at 10 35 16 AM" src="https://github.com/user-attachments/assets/b1ce5ac5-d78f-413d-a9c2-8df71d4959db" />

##### Random Forest 
* Accuracy: 100%
* Classification Report
<img width="453" alt="Screenshot 2025-05-02 at 10 35 52 AM" src="https://github.com/user-attachments/assets/239ce6f4-6abc-427c-99a5-75f75bcf3424" />

##### Random Forest with max_depth = 3
* Accuracy: 99.5%
* Classification Report
<img width="450" alt="Screenshot 2025-05-02 at 10 39 27 AM" src="https://github.com/user-attachments/assets/83829087-4232-4d4d-8af0-13f4b983a13b" />

* ROC curves not generated, but confusion matrices used to visualize class performance(in Notebook).
---

### Conclusions
* The PCOS dataset was clean, numerical, and well-structured, requiring minimal preprocessing. Key features like Menstrual Irregularity and Antral Follicle Count showed strong correlation with the diagnosis label, allowing even simple models to perform well.
* Both Logistic Regression and Random Forest models were trained and evaluated. Logistic Regression, as a baseline, achieved ~91.5% accuracy. The Random Forest achieved perfect performance (100% accuracy), raising overfitting concerns — which were addressed by **training a smaller Random Forest (3 trees, max depth 3)**. This smaller model still **achieved ~99.5% accuracy**, proving the dataset's high separability and the effectiveness of its features.
* 5-Fold Cross Validation showed strong consistency:
  * Random Forest: ~0.9975 mean accuracy
  * Logistic Regression: ~0.91 mean accuracy
* To improve interpretability, we **visualized a decision tree from the small forest, confirming Menstrual Irregularity and BMI as top decision features**.
* Overall, the project demonstrates how well-designed clinical features, even in a small dataset, can lead to highly accurate and interpretable machine learning models for health prediction tasks.

---

### Future Work
* Drop BMI for a Different Modeling Perspective: Given how tightly BMI is linked to PCOS, future experiments could completely exclude BMI from the feature set. This would help determine whether other features alone can provide robust predictions, and evaluate model dependence on correlated variables.

* Introduce Additional Models: Explore other ensemble learning techniques like
  * Gradient Boosting
  * XGBoost
  * LightGBM
  These may offer enhanced generalization and potentially better performance on edge cases.

* External Validation: To confirm generalizability, the current models can be evaluated on external clinical datasets with different patient distributions or additional biomarkers.

---

### Overview of Files in Repository

The project is organized into two folders: Final Notebooks and Scratch Notebooks.

Final Notebooks (4 files):
* Notebook.ipynb – Full project implementation: preprocessing, modeling, evaluation.
* df_head.txt – Output of the first few rows of the dataset.
* df_describe.txt – Summary stats like mean, min, max, etc.
* df_info.txt – Dataset structure: column types and null counts.

Scratch Notebooks (3 files):
* scratch-notebook1.ipynb – Initial data exploration and plotting.
* scratch-notebook2.ipynb – Trial runs, feature testing, model drafts.
* pcos_dataset.csv – Source dataset from Kaggle.
  
---

### How to Reproduce Results & Software Setup
* To reproduce the results, simply run Notebook.ipynb in the Final Notebooks folder. It walks through the full pipeline — from loading the data to cleaning, visualization, modeling, and evaluation.
* Everything runs from scratch — no pre-trained models or cached results.
* Setup Instructions
  1.Clone or download the repo.
  2.Open Notebook.ipynb in Jupyter Notebook or VS Code.
  3.Ensure the dataset pcos_dataset.csv (from Scratch Notebooks) is in the same folder or properly referenced.
* Required Libraries
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
    
* Install with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn

* Notes on Text Files: df_head.txt, df_info.txt, and df_describe.txt are included to let you preview the dataset structure without rerunning code.

---

### Citations
* Kaggle[Link](https://www.kaggle.com/datasets/samikshadalvi/pcos-diagnosis-dataset)
* PCOS-National Library of Medicine[Link](https://www.ncbi.nlm.nih.gov/books/NBK459251/)
  
