import streamlit as st
import pickle
import numpy as np
import pandas as pd

data = pd.read_csv('Video_Games_Sales_as_at_22_Dec_2016.csv')
original_data = data
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

with open('pickle/random_forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('pickle/svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)

with open('pickle/logistic_regression_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

with open('pickle/voting_classifier.pkl', 'rb') as f:
    voting_model = pickle.load(f)

with open('pickle/stacking_model.pkl', 'rb') as f:
    stacking_model = pickle.load(f)
with open('pickle/sajat_model.pkl', 'rb') as f:
    sajat = pickle.load(f)

# with open('pickle/encoder_developer.pkl', 'rb') as f:
#     encoder_developer = pickle.load(f)
# with open('pickle/encoder_platform.pkl', 'rb') as f:
#     encoder_platform = pickle.load(f)
# with open('pickle/encoder_publisher.pkl', 'rb') as f:
#     encoder_publisher = pickle.load(f)
# with open('pickle/encoder_rating.pkl', 'rb') as f:
#     encoder_rating = pickle.load(f)
# with open('pickle/encoder_y.pkl', 'rb') as f:
#     encoder_y = pickle.load(f)
with open('label_encoders.pkl','rb') as f:
    label_encoder = pickle.load(f)
from sklearn.preprocessing import RobustScaler
def preprocess_data(df):
    df
    df['Rating'] = label_encoder['Rating'].transform(df['Rating'])
    df['Platform'] = label_encoder['Platform'].transform(df['Platform'])
    df['Publisher'] = label_encoder['Publisher'].transform(df['Publisher'])
    df['Developer'] = label_encoder['Developer'].transform(df['Developer'])
    #df['Genre'] = encoder_y.transform(df['Genre'])
    scaler = RobustScaler()
    df = scaler.fit_transform(df)   

    return df

def predict_rf(input_data):

    predictions = rf_model.predict(input_data)
    genre_predictions = label_encoder['Genre'].inverse_transform(predictions)

    return genre_predictions

def predict_svm(input_data):
    predictions = svm_model.predict(input_data)
    genre_predictions = label_encoder['Genre'].inverse_transform(predictions)

    return genre_predictions

def predict_lr(input_data):
    predictions = lr_model.predict(input_data)
    genre_predictions = label_encoder['Genre'].inverse_transform(predictions)

    return genre_predictions

def predict_voting(input_data):
    predictions = voting_model.predict(input_data)
    genre_predictions = label_encoder['Genre'].inverse_transform(predictions)

    return genre_predictions

def predict_stacking(input_data):
    predictions = stacking_model.predict(input_data)
    genre_predictions = label_encoder['Genre'].inverse_transform(predictions)

    return genre_predictions
def predict_sajat(input_data):
    predictions = sajat.predict(input_data)
    genre_predictions = label_encoder['Genre'].inverse_transform(np.argmax(predictions, axis=1))

    return genre_predictions


# Streamlit app
def main():
    st.title('Video Game Genre Prediction')
    st.header('The original table')
    st.write(original_data)
    
    st.sidebar.title('Prediction')
    form = st.sidebar.form(key='input_form')
    
    platform_options = data['Platform'].unique()
    platform = form.selectbox('Platform', platform_options)

    year_of_release = form.number_input('Year of Release', min_value=1950, max_value=2023, step=1)

    publisher_options = data['Publisher'].unique()
    publisher = form.selectbox('Publisher', publisher_options)

    developer_options = data['Developer'].unique()
    developer = form.selectbox('Developer', developer_options)

    critic_score = form.number_input('Critic Score', min_value=0, max_value=100, step=1)

    user_score = form.number_input('User Score', min_value=0, max_value=100, step=1)

    rating_options = data['Rating'].unique()
    rating = form.selectbox('Rating', rating_options)

    na_sales = form.number_input('NA Sales', min_value=0, max_value=5000, step=100)

    eu_sales = form.number_input('EU Sales', min_value=0, max_value=5000, step=100)

    jp_sales = form.number_input('JP Sales', min_value=0, max_value=5000, step=100)

    other_sales = form.number_input('Other Sales', min_value=0, max_value=5000, step=100)

    critic_count = form.number_input('Critic count', min_value=0, max_value=100, step=1)

    user_count = form.number_input('User Count', min_value=0, max_value=100, step=1)

    submit_button = form.form_submit_button(label='Predict')
    if submit_button:
        global_sales=na_sales+eu_sales+other_sales+jp_sales
        # Create a DataFrame from the user input
        input_df = pd.DataFrame([[platform, year_of_release, publisher, na_sales,eu_sales,jp_sales,other_sales,global_sales, critic_score,critic_count, user_score,user_count,developer, rating]],
                                columns=['Platform', 'Year_of_Release', 'Publisher','NA_Sales','EU_Sales', 'JP_Sales','Other_Sales','Global_Sales', 'Critic_Score','Critic_Count', 'User_Score','User_Count', 'Developer','Rating'])

        # Make predictions using all models
        preprocessed_data = preprocess_data(input_df)
        rf_prediction = predict_rf(preprocessed_data)
        svm_prediction = predict_svm(preprocessed_data)
        lr_prediction = predict_lr(preprocessed_data)
        voting_prediction = predict_voting(preprocessed_data)
        stacking_prediction = predict_stacking(preprocessed_data)
        sajat_prediction = predict_sajat(preprocessed_data)


        st.header('Predictions')
        predictions={
            'Model': ['Random Forest', 'SVM', 'Logistic Regression', 'Voting Ensemble', 'Stacking','Sajat'],
            'Prediction': [rf_prediction[0], svm_prediction[0], lr_prediction[0], voting_prediction[0], stacking_prediction[0],sajat_prediction[0]]

        }
        st.table(predictions)

# Run the Streamlit app
if __name__ == '__main__':
    main()
