import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import categorical_accuracy
import pickle


model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('one_hotencoder_geo.pkl', 'rb') as f:
    one_hotencoder_geo = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


st.title('Customer Churn Prediction')

gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.number_input('Age', min_value=0, max_value=100)
tenure = st.number_input('Tenure', min_value=0, max_value=100)
balance = st.number_input('Balance', min_value=0.0)
products = st.number_input('Number of Products', min_value=0, max_value=10)
credit_score = st.number_input('Credit Score', min_value=0, max_value=1000)
active_member = st.selectbox('Active Member', [0,1])
estimated_salary = st.number_input('Estimated Salary', min_value=0.0)
has_cr_card = st.selectbox('Has Credit Card', [0,1])
geo = st.selectbox('Geography', ['France', 'Spain', 'Germany'])

if st.button('Predict'):
    gender_encoded = label_encoder_gender.transform([gender])
    geo_encoded = one_hotencoder_geo.transform([[geo]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded,columns=one_hotencoder_geo.get_feature_names_out(['Geography']))
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Geography': [geo_encoded_df],
        'Gender': [gender_encoded],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [active_member],
        'EstimatedSalary': [estimated_salary]
    })

    input_data_scaled = scaler.transform(input_data)

    prediction = model.predict(input_data_scaled)

    if prediction > 0.5:
        st.write('Customer will leave the bank')
    else:
        st.write('Customer will not leave the bank')

    st.write(prediction)

