import streamlit as st
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from fbprophet import Prophet


st.set_page_config(page_title='Industry Search')



def extract_columns(df):
    """Clean data and return useful columns"""

    # Remove white spaces
    df = df.applymap(lambda x: str(x).strip())

    # Calculate profit
    df['CP'] = df[' Unit Cost '].apply(lambda x: x.lstrip('$')).astype(float)
    df['SP'] = df[' Sales Price '].apply(lambda x: x.lstrip('$')).astype(float)
    df['Quantity'] = df['Quantity'].astype(int)
    df['Profit'] = (df['SP'] - df['CP']) * df['Quantity']

    # Clean data column
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Order Date'] = df['Order Date'].dt.tz_localize(None)

    return df[['Order Date', 'Profit']]



def resample_data(df):
    df.set_index('Order Date', inplace=True)
    df = df.resample('M').sum()

    return df



def perform_forecast(df):
    df_univar = df[["Order Date", "Profit"]].rename(columns={"Order Date":"ds", "Profit":"y"})
    
    # 70-30 Train-Test split
    train_size = int(df_univar.shape[0] * 0.7)
    df_train = df_univar.iloc[:train_size]
    df_test = df_univar.iloc[train_size:]
    assert df_univar.shape[0] == df_train.shape[0] + df_test.shape[0]

    # Fit model
    model = Prophet()
    model.fit(df_train)

    # Make future dates
    periods_to_predict = df_test.shape[0]
    future_dates = model.make_future_dataframe(periods=periods_to_predict, freq='M')
    assert df_univar.shape[0] == future_dates.shape[0]

    # Predict for the next n months
    forecast = model.predict(future_dates)

    return forecast






uploaded_file = st.file_uploader(label='Upload Sales Data', type=['csv', 'xlsx'])
uploaded_file = open('sample_sales_data3.csv')

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    
    with st.spinner('Extracting columns'):
        df = extract_columns(raw_df)

    df = resample_data(df)

    # Streamlit currently doesn't support the statsmodels library
    # Stationarity test
    # p_value = adfuller(df["Profit"])[1]
    # if p_value >= 0.05:
    #     st.error('Data is not stationary')

    

    df.reset_index(inplace=True)
    forecast = perform_forecast(df)

    st.dataframe(df)
    

