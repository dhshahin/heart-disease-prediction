[README.md](https://github.com/user-attachments/files/26331798/README.md)
# Heart Disease Prediction: End-to-End Machine Learning Pipeline

This project builds a complete machine learning pipeline to predict the presence of heart disease using clinical features. It covers data validation, preprocessing, model comparison, hyperparameter tuning, final evaluation, and reusable prediction scripts.

## Project Objective

The goal of this project is to develop a reliable binary classification model that predicts whether a patient is likely to have heart disease based on medical attributes such as age, chest pain type, cholesterol, maximum heart rate, and other clinical measurements.

## Problem Type

- **Task:** Binary Classification
- **Target Variable:** `target`
  - `1` = presence of heart disease
  - `0` = absence of heart disease

## Dataset

The dataset contains 303 patient records and 14 columns, including the target variable.

### Features
- age
- sex
- cp
- trestbps
- chol
- fbs
- restecg
- thalach
- exang
- oldpeak
- slope
- ca
- thal

### Notes
- No missing values were found
- One duplicate row was detected and removed
- The cleaned dataset used for modeling contains **302 rows**

## Project Workflow

1. Data loading and inspection
2. Missing-value and duplicate checks
3. Duplicate removal
4. Exploratory data analysis
5. Feature grouping into numeric and categorical variables
6. Preprocessing using `ColumnTransformer`
7. Baseline model training
8. Model comparison
9. Cross-validation
10. Hyperparameter tuning
11. Final evaluation
12. Model saving and reusable prediction pipeline

## Models Compared

The following models were evaluated:

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- K-Nearest Neighbors (KNN)

## Model Comparison Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|------|---------:|----------:|-------:|---:|--------:|
| SVM | 0.820 | 0.789 | 0.909 | 0.845 | 0.902 |
| Logistic Regression | 0.852 | 0.853 | 0.879 | 0.866 | 0.897 |
| Random Forest | 0.787 | 0.763 | 0.879 | 0.817 | 0.897 |
| KNN | 0.754 | 0.750 | 0.818 | 0.783 | 0.858 |

## Final Model Selection

The final selected model is **Logistic Regression**.

### Why Logistic Regression?
- Best overall balance between precision, recall, and F1-score
- Strong ROC-AUC performance
- More interpretable than more complex models
- Well suited for this dataset size

## Cross-Validation Performance

The baseline Logistic Regression model achieved:

- **Mean 5-fold CV ROC-AUC:** `0.908`
- **Standard Deviation:** `0.035`

After hyperparameter tuning, the best cross-validated ROC-AUC improved to:

- **Best CV ROC-AUC:** `0.919`

### Best Hyperparameters
- `C = 0.1`
- `penalty = l2`
- `solver = liblinear`

## Final Test Set Performance

| Metric | Value |
|------|------:|
| Accuracy | 0.852 |
| Precision | 0.833 |
| Recall | 0.909 |
| F1 Score | 0.870 |
| ROC-AUC | 0.894 |

## Confusion Matrix

```text
[[22  6]
 [ 3 30]]
```

This means the final model correctly identified:
- 22 true negatives
- 30 true positives

It made:
- 6 false positives
- 3 false negatives

## Key Interpretation

The final model achieved strong recall, meaning it successfully identified most heart disease cases in the test set. This is especially important in medical classification settings, where missing a positive case may be more costly than generating a false alarm.

## Important Features

Among the most influential features in the final Logistic Regression model were:

- `thal`
- `ca`
- `cp`
- `sex`
- `exang`
- `thalach`

A full feature importance table is available in:

```text
reports/model_results/feature_importance_logistic_regression.csv
```

## Project Structure

```text
heart-disease-prediction/
│
├── data/
│   ├── raw/
│   │   └── heart.csv
│   └── processed/
│       └── heart_clean.csv
│
├── models/
│   └── final_model.pkl
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   └── 02_modeling.ipynb
│
├── reports/
│   ├── figures/
│   │   └── roc_curve_final_logistic_regression.png
│   └── model_results/
│       ├── final_results_summary.csv
│       ├── model_comparison_results.csv
│       └── feature_importance_logistic_regression.csv
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/dhshahin/heart-disease-prediction.git
cd heart-disease-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python src/train.py
```

### 4. Run prediction script
```bash
python src/predict.py
```

## Example Prediction Output

The prediction script returns:
- predicted class
- probability of heart disease

Example:
```python
{'prediction': 0, 'probability_of_heart_disease': 0.0688}
```

## Tools and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Jupyter Notebook

## Limitations

- The dataset is relatively small
- This project is for educational and portfolio purposes
- The model should not be used for real clinical diagnosis without proper medical validation and external testing

## Future Improvements

- Add a Streamlit web app
- Add SHAP-based explainability
- Perform calibration analysis
- Test on additional heart disease datasets
- Package the project with unit tests and automated workflows

## Disclaimer

This project is intended for educational and demonstration purposes only. It is not a medical diagnostic tool and should not be used for real-world clinical decision-making.
