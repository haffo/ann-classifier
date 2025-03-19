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


model = tf.keras.models.load_model('artifacts/regression/model.h5')

with open('artifacts/regression/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('artifacts/regression/label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('artifacts/regression/one_hotencoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)

st.title('Salary Prediction')

gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.number_input('Age', min_value=1, max_value=100)
tenure = st.number_input('Tenure', min_value=1, max_value=100)
balance = st.number_input('Balance', min_value=1, max_value=1000000)
geo = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
num_of_products = st.number_input('Number of Products', min_value=1)
has_credit_card = st.selectbox('Has Credit Card', ['Yes', 'No'])
is_active_member = st.selectbox('Is Active Member', ['Yes', 'No'])
existed = st.number_input('Exited', min_value=0, max_value=1)
credit_score = st.number_input('Credit Score', min_value=1, max_value=850)


if st.button('Predict'):
    gender_encoded = label_encoder_gender.transform([gender])

    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [gender_encoded],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [1 if has_credit_card == 'Yes' else 0],
        'IsActiveMember': [1 if is_active_member == 'Yes' else 0],
        'Exited': [existed]
    })

        # One-hot encode 'Geography'
    geo_encoded = onehot_encoder_geo.transform([[geo]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

    # Combine one-hot encoded columns with input data
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    input_data_scaled = scaler.transform(input_data)

    prediction = model.predict(input_data_scaled)    

    st.write('Predicted Salary:', prediction[0][0])

