import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

data = pd.read_csv('Video_Games_Sales_as_at_22_Dec_2016.csv')
data = data.dropna()

data['Year_of_Release'] = data['Year_of_Release'].astype(int)
data['User_Count'] = data['User_Count'].astype(int)
data['User_Score'] = data['User_Score'].mul(10).astype(int)
data['Critic_Count'] = data['Critic_Count'].astype(int)
data['Critic_Score'] = data['Critic_Score'].astype(int)
data['NA_Sales'] = data['NA_Sales'].mul(100).astype(int)
data['EU_Sales'] = data['EU_Sales'].mul(100).astype(int)
data['Global_Sales'] = data['Global_Sales'].mul(100).astype(int)

data = data[data['Global_Sales'] >= 0.2]
data = data.drop(['Other_Sales'], axis=1)
data = data.drop(['JP_Sales'], axis=1)
data = data.drop(['Name'], axis=1)

with open('random_forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)

with open('logistic_regression_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

with open('voting_classifier.pkl', 'rb') as f:
    voting_model = pickle.load(f)

with open('stacking_model.pkl', 'rb') as f:
    stacking_model = pickle.load(f)

with open('Ada_classifier.pkl', 'rb') as f:
    adaboosting_model = pickle.load(f)

with open('encoder_developer.pkl', 'rb') as f:
    encoder_developer = pickle.load(f)
with open('encoder_platform.pkl', 'rb') as f:
    encoder_platform = pickle.load(f)
with open('encoder_publisher.pkl', 'rb') as f:
    encoder_publisher = pickle.load(f)
with open('encoder_rating.pkl', 'rb') as f:
    encoder_rating = pickle.load(f)
with open('encoder_y.pkl', 'rb') as f:
    encoder_y = pickle.load(f)

def preprocess_data(df):
    out_df = df
    out_df['Rating'] = encoder_rating.transform(out_df['Rating'])
    out_df['Platform'] = encoder_platform.transform(out_df['Platform'])
    out_df['Publisher'] = encoder_publisher.transform(out_df['Publisher'])
    out_df['Developer'] = encoder_developer.transform(out_df['Developer'])
    #df['Genre'] = encoder_y.transform(df['Genre'])
    
    return out_df

def predict_rf(input_data):
    st.write("RF")
    st.write(input_data)

    input_data = preprocess_data(input_data)
    predictions = rf_model.predict(input_data)
    genre_predictions = encoder_y.inverse_transform(predictions)
    
    st.write("after RF")
    st.write(input_data)
    return genre_predictions

def predict_svm(input_data):
    st.write("SVM")
    st.write(input_data)
    input_data = preprocess_data(input_data)
    predictions = svm_model.predict(input_data)
    genre_predictions = encoder_y.inverse_transform(predictions)

    return genre_predictions

def predict_lr(input_data):
    input_data = preprocess_data(input_data)
    predictions = lr_model.predict(input_data)
    genre_predictions = encoder_y.inverse_transform(predictions)

    return genre_predictions

def predict_voting(input_data):
    input_data = preprocess_data(input_data)
    predictions = voting_model.predict(input_data)
    genre_predictions = encoder_y.inverse_transform(predictions)

    return genre_predictions

def predict_stacking(input_data):
    input_data = preprocess_data(input_data)
    predictions = stacking_model.predict(input_data)
    genre_predictions = encoder_y.inverse_transform(predictions)

    return genre_predictions

def predict_ada(input_data):
    input_data = preprocess_data(input_data)
    predictions = adaboosting_model.predict(input_data)
    genre_predictions = encoder_y.inverse_transform(predictions)

    return genre_predictions

# Streamlit app
def main():
    st.title('Video Game Genre Prediction')

    # Create a form for user input
    st.header('Input Data')
    form = st.form(key='input_form')

    platform = form.selectbox('Platform', data['Platform'].unique())
    year_of_release = form.number_input('Year of Release', min_value=1950, max_value=2023, step=1)
    #genre = form.selectbox('Genre', data['Genre'].unique())
    publisher = form.selectbox('Publisher', data['Publisher'].unique())
    developer = form.selectbox('Developer', data['Developer'].unique())
    critic_score = form.number_input('Critic Score', min_value=0, max_value=100, step=1)
    user_score = form.number_input('User Score', min_value=0.0, max_value=10.0, step=0.1)
    rating = form.selectbox('Rating', data['Rating'].unique())
    na_sales = form.number_input('NA Sales',min_value=0, max_value=5000, step=100)
    eu_sales = form.number_input('EU Sales',min_value=0, max_value=5000, step=100)
    critic_count = form.number_input('Cricic count',min_value=0, max_value=100, step=1)
    user_count = form.number_input('User Count',min_value=0, max_value=100, step=1)
    submit_button = form.form_submit_button(label='Predict')

    if submit_button:
        global_sales=na_sales+eu_sales
        # Create a DataFrame from the user input
        input_df = pd.DataFrame([[platform, year_of_release, publisher, na_sales,eu_sales,global_sales, critic_score,critic_count, user_score,user_count,developer, rating]],
                                columns=['Platform', 'Year_of_Release', 'Publisher','NA_Sales','EU_Sales', 'Global_Sales', 'Critic_Score','Critic_Count', 'User_Score','User_Count', 'Developer','Rating'])

        # Make predictions using all models

        rf_prediction = predict_rf(input_df)
        svm_prediction = predict_svm(input_df)
        lr_prediction = predict_lr(input_df)

        st.header('Predictions')

        st.subheader('Random Forest Classifier Prediction')
        st.write(rf_prediction)

        st.subheader('SVM Classifier Prediction')
        st.write(svm_prediction)

        st.subheader('Logistic Regression Classifier Prediction')
        st.write(lr_prediction)

# Run the Streamlit app
if __name__ == '__main__':
    main()
