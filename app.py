import joblib
import streamlit as st 
import pandas as pd

model = joblib.load('model_svm.pkl')
scaler = joblib.load('StandardScaler.pkl')
expected_columns = joblib.load('expected_columns.pkl')

st.title('Loan Approval Prediction')
st.markdown('Provide the following details:')

gender = st.selectbox('Gender', [0, 1])
st.markdown('0 is for female and 1 is for male')
age = st.number_input('Age', 18, 70, 20)
income = st.number_input('Income', 10000, 300000, 50000)
credit_score = st.slider('Credit Score', 300, 850, 580)
years_experience = st.slider('Experience (in years)', 1, 40, 5)
loan_amount = st.number_input('Loan Amount', 1000, 1000000, 50000)
employment_type = st.selectbox('Employment Type', ['Salaried', 'Self-Employed', 'Unemployed'])
education = st.selectbox('Educational Level', ['High School', 'Bachelors', 'Masters', 'PhD'])

if st.button('Predict'):

    input_df = pd.DataFrame({
        'Age': [age],
        'Income': [income],
        'LoanAmount': [loan_amount],
        'CreditScore': [credit_score],
        'YearsExperience': [years_experience],
        'Is_Male': [gender]
    })

    numerical_cols = [
        'Age',
        'Income',
        'LoanAmount',
        'CreditScore',
        'YearsExperience'
    ]

    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    input_df['EmploymentType_Salaried'] = 0
    input_df['EmploymentType_Self-Employed'] = 0
    input_df['EmploymentType_Unemployed'] = 0

    input_df['Education_Bachelors'] = 0
    input_df['Education_High School'] = 0
    input_df['Education_Masters'] = 0
    input_df['Education_PhD'] = 0

    input_df['EmploymentType_' + employment_type] = 1
    input_df['Education_' + education] = 1

    input_df = input_df[expected_columns]

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.success('Loan Approved')
    else:
        st.error('Loan Rejected')   