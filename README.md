### 1.HCV Stage Prediction (HCV_Research.ipynb)

This project predicts the stages of Hepatitis C Virus (HCV) infection Blood Donor, Suspect Blood Donor, Hepatitis, Fibrosis, and Cirrhosis — using clinical and biochemical parameters from the Hepatitis C Prediction dataset. The study implements both Machine Learning and Deep Learning techniques for accurate disease staging.

**Features:**

-**Data Preprocessing:**
Missing value handling, duplicate removal, irrelevant column drop (Unnamed: 0), label encoding, one-hot encoding, and class balancing using SMOTE.

-**Models Used:**
-Random Forest Classifier
-Stacking Classifier
-Bagging Classifier
-Voting Classifier
-Feed Forward Neural Network (FNN)

-**Ensemble Techniques:**
Bagging, Voting, and Stacking (with Random Forest and SVC as base estimators).

-**Data Transformation:**
StandardScaler applied for normalization; class imbalance treated with SMOTE to improve generalization.

-**Exploratory Data Analysis (EDA):**
Statistical summaries, distribution plots, correlation heatmaps, and pair plots to identify relationships between clinical parameters.

-**Evaluation Metrics:**
Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.

-**Final Result:**
The Stacking Classifier achieved the highest accuracy of 94.07%, outperforming Random Forest and other models.
