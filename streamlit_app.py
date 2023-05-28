import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the preprocessed data
data = pd.read_csv('preprocessed_data.csv')

# Load the trained models
with open('random_forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)

with open('logistic_regression_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

fnn_model = load_model('fnn_model.h5')

# Load the label encoder
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Function to preprocess the input data
def preprocess_data(df):
    # Perform the same preprocessing steps as in the notebook
    df['Rating'] = label_encoder.transform(df['Rating'])
    df['Platform'] = label_encoder.transform(df['Platform'])
    df['Publisher'] = label_encoder.transform(df['Publisher'])
    df['Developer'] = label_encoder.transform(df['Developer'])

    # Standardize the features
    scaler = StandardScaler()
    df = scaler.fit_transform(df)

    return df

# Function to make predictions using the RandomForest model
def predict_rf(input_data):
    # Preprocess the input data
    input_data = preprocess_data(input_data)

    # Make predictions
    predictions = rf_model.predict(input_data)

    # Convert predictions back to original labels
    genre_predictions = label_encoder.inverse_transform(predictions)

    return genre_predictions

# Function to make predictions using the SVM model
def predict_svm(input_data):
    # Preprocess the input data
    input_data = preprocess_data(input_data)

    # Make predictions
    predictions = svm_model.predict(input_data)

    # Convert predictions back to original labels
    genre_predictions = label_encoder.inverse_transform(predictions)

    return genre_predictions

# Function to make predictions using the Logistic Regression model
def predict_lr(input_data):
    # Preprocess the input data
    input_data = preprocess_data(input_data)

    # Make predictions
    predictions = lr_model.predict(input_data)

    # Convert predictions back to original labels
    genre_predictions = label_encoder.inverse_transform(predictions)

    return genre_predictions

# Function to make predictions using the FNN model
def predict_fnn(input_data):
    # Preprocess the input data
    input_data = preprocess_data(input_data)

    # Make predictions
    predictions = fnn_model.predict(input_data)

    # Convert predictions to genre labels
    genre_predictions = label_encoder.inverse_transform(predictions.argmax(axis=1))

    return genre_predictions

# Streamlit app
def main():
    st.title('Video Game Genre Prediction')

    # Create a form for user input
    st.header('Input Data')
    form = st.form(key='input_form')

    name = form.text_input('Name')
    platform = form.selectbox('Platform', data['Platform'].unique())
    year_of_release = form.number_input('Year of Release', min_value=1950, max_value=2023, step=1)
    genre = form.selectbox('Genre', data['Genre'].unique())
    publisher = form.selectbox('Publisher', data['Publisher'].unique())
    developer = form.selectbox('Developer', data['Developer'].unique())
    critic_score = form.number_input('Critic Score', min_value=0, max_value=100, step=1)
    user_score = form.number_input('User Score', min_value=0.0, max_value=10.0, step=0.1)
    rating = form.selectbox('Rating', data['Rating'].unique())

    submit_button = form.form_submit_button(label='Predict')

    if submit_button:
        # Create a DataFrame from the user input
        input_df = pd.DataFrame([[name, platform, year_of_release, genre, publisher, developer, critic_score, user_score, rating]],
                                columns=['Name', 'Platform', 'Year_of_Release', 'Genre', 'Publisher', 'Developer', 'Critic_Score', 'User_Score', 'Rating'])

        # Make predictions using all models
        rf_prediction = predict_rf(input_df)
        svm_prediction = predict_svm(input_df)
        lr_prediction = predict_lr(input_df)
        fnn_prediction = predict_fnn(input_df)

        st.header('Predictions')

        st.subheader('Random Forest Classifier Prediction')
        st.write(rf_prediction)

        st.subheader('SVM Classifier Prediction')
        st.write(svm_prediction)

        st.subheader('Logistic Regression Classifier Prediction')
        st.write(lr_prediction)

        st.subheader('FNN Classifier Prediction')
        st.write(fnn_prediction)

# Run the Streamlit app
if __name__ == '__main__':
    main()
