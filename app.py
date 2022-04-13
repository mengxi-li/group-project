import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
from datetime import date
import styles

# Start date refers to the date from which we're reading the histical data
start = '2015-01-01'
end = date.today().strftime("%Y-%m-%d")

# Setting page styles
st.set_page_config(
     page_title="COMP 377 | Group 6",
    #  layout="wide",   
 )
st.title("Predictr - Stock Prediction App")


# Hiding default styles provided by Streamlit
hide_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """

st.markdown(hide_style, unsafe_allow_html=True)
st.markdown(styles.footer,unsafe_allow_html=True)

user_input = st.text_input('Enter Stock Ticker','AMZN')
df = data.DataReader(user_input,'yahoo',start,end)

# UI Spinner lets the user know that the prediction is still taking place
with st.spinner(f'Predicting "{user_input}" stock'):
    # Data received from Yahoo Finance - Displayed as a table
    st.subheader('Data from 2015-2022')
    st.write(df.describe())

    # Closing price chart - Displayed as a graph
    st.subheader('Closing Price Chart')
    fig = plt.figure(figsize = (12,6))
    plt.plot(df.Close)
    st.pyplot(fig)

    # Closing price chart vs Time chart with 50 days moving average applied on it - Displayed as a graph
    st.subheader('Closing Price vs Time Chart with 50MA')
    ma50 = df.Close.rolling(50).mean()
    fig = plt.figure(figsize = (12,6))
    plt.plot(ma50)
    plt.plot(df.Close)
    st.pyplot(fig)

    # Closing price chart vs Time chart with 50 and 100 days moving average applied on it - Displayed as a graph
    st.subheader('Closing Price vs Time Chart with 50MA&100MA')
    ma50 = df.Close.rolling(50).mean()
    ma100 = df.Close.rolling(100).mean()
    fig = plt.figure(figsize = (12,6))
    plt.plot(ma100)
    plt.plot(ma50)
    plt.plot(df.Close)
    st.pyplot(fig)

    # Splitting 70:30 for training and testing
    data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

    from sklearn.preprocessing import MinMaxScaler

    # Feature range (0, 1)
    scaler = MinMaxScaler(feature_range=(0,1))

    # Model fit transform
    data_training_array = scaler.fit_transform(data_training)

    model = load_model('keras_model.h5')

    # Retrieve last 100 days data
    past_100_days = data_training.tail(100)
    final_df = past_100_days.append(data_testing, ignore_index=True)
    
    # Model fit transform
    input_data = scaler.fit_transform(final_df)

    # Initializing the test and prediction variables
    x_test = []
    y_test = []
    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i,0])

    x_test,y_test = np.array(x_test),np.array(y_test)
    y_predicted = model.predict(x_test)
    scaler=scaler.scale_

    # y_pred value is calculated by multiplying the x_test nparray predicted value with the scaler
    scale_factor = 1/scaler[0]
    y_predicted = y_predicted*scale_factor
    y_test = y_test*scale_factor

    # Original vs Predicted graph
    st.subheader('Predictions vs Original')
    fig2 = plt.figure(figsize=(12,6))
    plt.plot(y_test,'b', label='original price')
    plt.plot(y_predicted,'r', label='predicted price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)
st.success(f'Stock prediction for {user_input} is succesful')