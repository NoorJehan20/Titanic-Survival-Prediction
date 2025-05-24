import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import shap
import joblib

import sklearn
print(sklearn.__version__)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

#Loading Datasets
train_df = pd.read_csv("dataset/train.csv")
test_df = pd.read_csv("dataset/test.csv")
submission_df = pd.read_csv("dataset/gender_submission.csv")

#Combining datasets for feature engineering
data = pd.concat([train_df, test_df], sort=False)

#Feature engineering
data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)
data['Title'] = data['Title'].replace(['Mlle', 'Ms'], 'Miss')
data['Title'] = data['Title'].replace(['Mme', 'Lady', 'Countess', 'Dona'], 'Mrs')
data['Title'] = data['Title'].replace(['Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer'], 'Rare')

data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
data['IsAlone'] = (data['FamilySize'] == 1).astype(int)

data['Fare'] = data['Fare'].fillna(data['Fare'].median())
data['Embarked'] = data['Embarked'].fillna('S')
data['Age'] = data['Age'].fillna(data['Age'].median())

#Encoding categorical variables
data = pd.get_dummies(data, columns=['Sex', 'Embarked', 'Title'], drop_first=True)

#Dropping unnecessary columns
data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

#Separate train and test sets
train_cleaned = data[:len(train_df)]
test_cleaned = data[len(train_df):]
X = train_cleaned.drop("Survived", axis=1)
y = train_cleaned["Survived"]
X_test_final = test_cleaned.drop("Survived", axis=1)

#Splitting for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#Training XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_val)
print("XGBoost Accuracy:", accuracy_score(y_val, y_pred))

#Feature_Importances
plt.figure(figsize=(12, 6))
importances = xgb_model.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=90)
plt.title("Feature Importances - XGBoost")
plt.tight_layout()
plt.show()

#GridSearchCV
params = {
    'max_depth': [3, 4],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [100, 150],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
grid_search = GridSearchCV(estimator=XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
param_grid=params, cv=5, n_jobs=-1, verbose=1, scoring='accuracy')
grid_search.fit(X, y)
print("Best Parameters:", grid_search.best_params_)

#SHAP Explainability
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_val)
shap.summary_plot(shap_values, X_val)

#Ensembling with Voting Classifier
log_clf = LogisticRegression(max_iter=1000, random_state=42)
rf_clf = RandomForestClassifier()
voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rf_clf), ('xgb', xgb_model)], voting='soft')
voting_clf.fit(X_train, y_train)
y_pred_voting = voting_clf.predict(X_val)
print("Voting Classifier Accuracy:", accuracy_score(y_val, y_pred_voting))

#Final predictions for submission
final_preds = voting_clf.predict(X_test_final)
submission_df['Survived'] = final_preds
submission_df.to_csv("submission.csv", index=False)

#Saving model for Streamlit
joblib.dump(voting_clf, "titanic_model.pkl")