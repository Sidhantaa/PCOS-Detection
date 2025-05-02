![](UTA-DataScience-Logo.png)

# PCOS Detection using Binary ML Models
This project uses machine learning models to predict whether a patient has PCOS or not using clinical features from a Kaggle tabular dataset.  

---

## Overview
The goal of this project is to predict whether a patient has Polycystic Ovary Syndrome (PCOS) using clinical features and supervised machine learning. The dataset, sourced from Kaggle, includes 1,000 patient records with numerical features such as Age, BMI, Testosterone Level, Antral Follicle Count, and a binary feature for Menstrual Irregularity. The target variable is PCOS_Diagnosis (1 = PCOS, 0 = No PCOS), making this a binary classification task. This project investigates how well these features predict PCOS and how interpretable, reliable models can be built for healthcare applications. While features like Menstrual Irregularity and Antral Follicle Count showed strong correlations with PCOS, others such as BMI were less informative, prompting further experiments without them.

Two models were developed and tested:
* Logistic Regression — selected for its simplicity and interpretability
* Random Forest — chosen for its ability to model non-linear relationships
  
* The dataset was split 80/20 into training and testing sets. Both models were evaluated using accuracy, precision, recall, F1 score, ROC-AUC, and confusion matrices.
* Additionally, 5-fold cross-validation was performed to verify generalizability. Logistic Regression achieved ~91% accuracy, correctly identifying most PCOS cases — a key goal in clinical diagnosis.
* The Random Forest classifier achieved 100% accuracy on the test set, which raised concerns about overfitting.To investigate this, a smaller Random Forest model with just 3 trees and a depth of 3 was trained.It still achieved ~99.5% accuracy, confirming that strong signals exist in the data and the classes are well-separated.
* Further steps included visualizing feature correlations, interpreting decision trees, generating confusion matrices, and creating bar and histogram plots to demonstrate class separation.
* This project shows that with proper preprocessing and analysis, a small clinical dataset can be effectively modeled for disease prediction using simple, interpretable machine learning workflows.

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

Graph 1.1: Distribution of Numerical Features in the Dataset 
<img width="1113" alt="Screenshot 2025-05-02 at 11 15 18 AM" src="https://github.com/user-attachments/assets/8eb907ee-6e0c-4dc8-9902-b842233fc98b" />


Graph 1.2: Distirbution of Numerication Features by PCOS Class
<img width="1114" alt="Screenshot 2025-05-02 at 11 16 11 AM" src="https://github.com/user-attachments/assets/77d86f84-601b-4061-a0e6-7fb2dbdae4b0" />

### Problem Formulation
* The PCOS classification task is formulated as a supervised binary classification problem, where the input features consist of clinical measurements — Age, BMI, Menstrual Irregularity (binary), Testosterone Level, and Antral Follicle Count — and the output is a binary label (PCOS Diagnosis), where 1 represents a confirmed diagnosis of PCOS and 0 represents no diagnosis.
* Since the dataset is fully labeled and relatively clean, no imputation or encoding was necessary. The data structure allowed us to directly apply classification models without extensive preprocessing.
* Two models were implemented and evaluated:
  * Logistic Regression: Chosen as a baseline model due to its simplicity, interpretability, and effectiveness on binary classification tasks.
  * Random Forest Classifier: Used for its ability to handle non-linear relationships, robustness, and automatic feature selection.
* In addition to these models, a smaller Random Forest (with fewer trees and limited depth) was trained to test whether a simpler model could perform comparably. This helped assess overfitting concerns.
* To evaluate model generalization, we also applied 5-fold cross-validation on both classifiers and measured standard metrics including accuracy, precision, recall, F1-score, and confusion matrices.

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

##### Logistic regression
* Accuracy: 91%
* Classification Report
<img width="460" alt="Screenshot 2025-05-02 at 10 35 16 AM" src="https://github.com/user-attachments/assets/b1ce5ac5-d78f-413d-a9c2-8df71d4959db" />

###### Random Forest 
* Accuracy: 100%
* Classification Report
<img width="453" alt="Screenshot 2025-05-02 at 10 35 52 AM" src="https://github.com/user-attachments/assets/239ce6f4-6abc-427c-99a5-75f75bcf3424" />

###### Random Forest with max_depth = 3
* Accuracy: 99.5%
* Classification Report
<img width="450" alt="Screenshot 2025-05-02 at 10 39 27 AM" src="https://github.com/user-attachments/assets/83829087-4232-4d4d-8af0-13f4b983a13b" />

* ROC curves not generated, but confusion matrices used to visualize class performance.
---

### Conclusions
* The PCOS dataset was clean, numerical, and well-structured, requiring minimal preprocessing. Key features like Menstrual Irregularity and Antral Follicle Count showed strong correlation with the diagnosis label, allowing even simple models to perform well.
* Both Logistic Regression and Random Forest models were trained and evaluated. Logistic Regression, as a baseline, achieved ~91.5% accuracy. The Random Forest achieved perfect performance (100% accuracy), raising overfitting concerns — which were addressed by training a smaller Random Forest (3 trees, max depth 3). This smaller model still achieved ~99.5% accuracy, proving the dataset's high separability and the effectiveness of its features.
* 5-Fold Cross Validation showed strong consistency:
  * Random Forest: ~0.9975 mean accuracy
  * Logistic Regression: ~0.91 mean accuracy
* To improve interpretability, we visualized a decision tree from the small forest, confirming Menstrual Irregularity and BMI as top decision features.
* Overall, the project demonstrates how well-designed clinical features, even in a small dataset, can lead to highly accurate and interpretable machine learning models for health prediction tasks.

---

### Future Work
Drop BMI for a Different Modeling Perspective: Given how tightly BMI is linked to PCOS, future experiments could completely exclude BMI from the feature set. This would help determine whether other features alone can provide robust predictions, and evaluate model dependence on correlated variables.

Introduce Additional Models: Explore other ensemble learning techniques like
* Gradient Boosting
* XGBoost
* LightGBM
These may offer enhanced generalization and potentially better performance on edge cases.

Feature Importance Techniques: Use permutation importance, SHAP values, or model-based feature selection to quantitatively determine the contribution of each feature, beyond what decision trees visually suggest.

External Validation: To confirm generalizability, the current models can be evaluated on external clinical datasets with different patient distributions or additional biomarkers, simulating real-world application better.

Automated Hyperparameter Tuning: GridSearchCV or RandomizedSearchCV could be introduced for both Logistic Regression and Random Forest to systematically find optimal model configurations instead of using fixed parameters.

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

### Citations
* Kaggle[Link](https://www.kaggle.com/datasets/samikshadalvi/pcos-diagnosis-dataset)
* PCOS-National Library of Medicine[Link](https://www.ncbi.nlm.nih.gov/books/NBK459251/)
  
