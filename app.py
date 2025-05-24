import streamlit as st
import pandas as pd
import joblib

#Loading trained model
model = joblib.load("titanic_model.pkl")

st.set_page_config(page_title="Titanic Survival Prediction")
st.title("🚢 Titanic Survival Prediction App")

#Input fields
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare", value=30.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])
title = st.selectbox("Title", ["Mr", "Miss", "Mrs", "Master", "Rare"])

#Feature engineering like in training
family_size = sibsp + parch + 1
is_alone = int(family_size == 1)

#One-hot encoding for inputs
input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'FamilySize': [family_size],
    'IsAlone': [is_alone],
    'Sex_male': [1 if sex == 'male' else 0],
    'Embarked_Q': [1 if embarked == 'Q' else 0],
    'Embarked_S': [1 if embarked == 'S' else 0],
    'Title_Miss': [1 if title == 'Miss' else 0],
    'Title_Mr': [1 if title == 'Mr' else 0],
    'Title_Mrs': [1 if title == 'Mrs' else 0],
    'Title_Rare': [1 if title == 'Rare' else 0]
})

#Prediction
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    st.success("🎉 Survived!" if prediction == 1 else "❌ Did not survive.")
