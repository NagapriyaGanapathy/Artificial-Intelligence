### 1. HCV Stage Prediction (HCV_Research.ipynb)

This project predicts the stages of Hepatitis C Virus infection Blood Donor, Suspect Blood Donor, Hepatitis, Fibrosis, and Cirrhosis  using clinical and biochemical parameters from the Hepatitis C Prediction dataset.  
The study implements both Machine Learning and Deep Learning techniques for accurate disease staging.

**Features:**

- **Data Preprocessing:** Missing value handling, duplicate removal, irrelevant column drop, label encoding, one-hot encoding, and class balancing using SMOTE.
  
- **Models Used:**
  - Random Forest Classifier  
  - Stacking Classifier  
  - Bagging Classifier  
  - Voting Classifier  
  - Feed Forward Neural Network
  
- **Ensemble Techniques:** Bagging, Voting, and Stacking (with Random Forest and SVC as base estimators).
  
- **Data Transformation:** StandardScaler applied for normalization; class imbalance treated with SMOTE to improve generalization.
  
- **Exploratory Data Analysis (EDA):** Statistical summaries, distribution plots, correlation heatmaps, and pair plots to identify relationships between clinical parameters.
  
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.

**Final Result:**

The **Stacking Classifier** achieved the highest accuracy of **94.07%**, outperforming Random Forest and other models.

---

### 2. Adult Income Prediction (CA2ML_Individual.ipynb)

This project predicts whether a person’s annual income exceeds $50,000 using demographic and employment data from the Adult dataset.  
It applies Logistic Regression, Decision Tree, and Random Forest models, along with ensemble methods and PCA for improved performance.

**Overview:**
- Dataset: 32,561 records with demographic & work attributes  
- Target: Income level (>50K or <=50K)
- Preprocessing: Missing value handling, encoding, feature selection, and scaling  

**Models & Techniques:**
- **Logistic Regression** – best performing model (~84% accuracy)  
- **Decision Tree & Random Forest** – tested with cross-validation  
- **Ensemble Methods:** Bagging, Boosting (AdaBoost), and Stacking  
- **PCA Feature Extraction:** Improved Logistic Regression accuracy (from 80% → 84%)  

**Results Summary:**

| Model | Feature Extraction | Accuracy |
|--------|--------------------|-----------|
| Logistic Regression | With PCA | **84%** |
| Logistic Regression | Without PCA | 80% |

**Conclusion:**
Logistic Regression with PCA provided the best and most interpretable results.  
The study highlights how ensemble and feature extraction techniques enhance model robustness and accuracy.

---

### 3. Credit Card Fraud Detection (Fraud_detection.ipynb)

This project detects fraudulent credit card transactions using machine learning.  
The dataset has 555,719 records with 22 features and no missing values.

**Target:** is_fraud → (1 = Fraudulent, 0 = Legitimate)

**Models Used:**
- Logistic Regression  
- Decision Tree  
- Random Forest  
- Gaussian Naive Bayes  

**Process:**
- Fixed data leakage by encoding after train-test split  
- Improved accuracy and recall with better sampling and feature handling  
- Removed unhelpful features and retained key outlier patterns

**Result:**  
**Random Forest Classifier** performed best with strong accuracy and recall.


