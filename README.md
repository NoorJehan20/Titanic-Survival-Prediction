# 🚢 Titanic Survival Prediction

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![GitHub stars](https://img.shields.io/github/stars/NoorJehan20/Titanic-Survival-Prediction?style=social)

This project predicts passenger survival on the Titanic using machine learning models. It is built for the Kaggle "Titanic - Machine Learning from Disaster" competition.

## 📌 Project Overview

- Data preprocessing with feature engineering:
  - Extracted titles from passenger names
  - Created family size and isolation features
  - Handled missing values and encoded categorical variables
- Trained models:
  - XGBoost Classifier
  - Random Forest Classifier
  - Logistic Regression
- Ensemble learning with Voting Classifier for improved accuracy
- Hyperparameter tuning using GridSearchCV
- Model explainability with SHAP plots
- Generated submission file for Kaggle competition
- Streamlit app for interactive prediction (runs locally)

## 📁 Dataset

- Source: [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data)  
- Files used:
  - `train.csv` (training data)
  - `test.csv` (test data for prediction)
  - `gender_submission.csv` (sample submission format)

## 🧪 Installation & Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/NoorJehan20/Titanic-Survival-Prediction.git
   cd Titanic-Survival-Prediction
   ````

2. (Recommended) Create and activate a virtual environment:

   ```bash
   python -m venv venv
   # Linux/macOS
   source venv/bin/activate
   # Windows
   .\venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## 🚀 How to Run Locally

* To train and evaluate the model:

  ```bash
  python titanic_model.py
  ```

* To launch the Streamlit web app locally for interactive predictions:

  ```bash
  streamlit run app.py
  ```

* After running, open the URL shown in the terminal (usually [http://localhost:8501](http://localhost:8501)) in your web browser.

> **Note:** The Streamlit app is designed for local use only and is not deployed online.

## 📊 Results & Visualizations

* Model accuracy and classification reports
* Feature importance visualization with XGBoost
* SHAP explainability plots for interpretability
* Submission file (`submission.csv`) ready for Kaggle upload

## 📂 File Structure

```
├── app.py                 # Streamlit web app for local use
├── titanic_model.py       # Training, evaluation, and prediction scripts
├── dataset/               # Dataset files (download from Kaggle)
│   ├── train.csv
│   ├── test.csv
│   └── gender_submission.csv
├── submission.csv         # Kaggle submission predictions
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## 🙌 Acknowledgments

* Kaggle Titanic Competition for dataset and challenge
* SHAP library for model explainability
* XGBoost, scikit-learn, and Streamlit libraries

## 👩‍💻 Author

**Noor Jehan**
[GitHub Profile](https://github.com/NoorJehan20) | [LinkedIn](www.linkedin.com/in/noor-jehan-5a4161278)
