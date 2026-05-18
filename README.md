# An Interpretable Hybrid ML Framework for Early Prediction of PCOS

## Overview
This project predicts Polycystic Ovary Syndrome (PCOS) using a hybrid machine learning framework.

The project combines:
- Random Forest
- XGBoost
- Logistic Regression
- Explainable AI concepts

## Dataset
Dataset used:
PCOS clinical dataset with 541 patient records and 45 features.

## Features
- BMI
- Weight
- Height
- Hormonal levels
- Follicle count
- Skin darkening
- Hair growth
- Weight gain
- Lifestyle features

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Matplotlib
- Seaborn

## Machine Learning Models
- Logistic Regression
- Random Forest
- XGBoost
- Voting Classifier

## Run the Project

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train the model
```bash
python src/train_model.py
```

### Run Streamlit App
```bash
streamlit run app/app.py
```

## Expected Accuracy
Hybrid model accuracy: 93% to 96%

## Author
Samyuktha
