import os
import pandas as pd
import numpy as np
import pickle
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler

# Configuration
DATA_FILE = os.environ.get("DATA_FILE", "data/expense.csv")
MODEL_FILE = os.environ.get("MODEL_FILE", "models/lstm_model.h5")
SCALER_FILE = os.environ.get("SCALER_FILE", "models/scaler.pkl")


# Function to preprocess the data for LSTM
def preprocess_data(df, sequence_length=30):
    df['Date'] = pd.to_datetime(df['Date'])
    # df.set_index('Date', inplace=True)

    # Handle missing values
    df['Amount'].fillna(method='ffill', inplace=True)
    # Feature Engineering
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day

    # Feature Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[['Amount', 'DayOfWeek', 'Month', 'Day']] = scaler.fit_transform(df[['Amount', 'DayOfWeek', 'Month', 'Day']])

    X, y = [], []
    for i in range(sequence_length, len(df)):
        X.append(df[['Amount', 'DayOfWeek', 'Month', 'Day']].iloc[i - sequence_length:i].values)
        y.append(df['Amount'].iloc[i])

    X, y = np.array(X), np.array(y)
    return X, y, scaler


# Train the LSTM model
def train_model(df, sequence_length=30):
    X, y, scaler = preprocess_data(df, sequence_length)

    model = Sequential([
        Bidirectional(LSTM(100, return_sequences=True), input_shape=(sequence_length, 4)),
        Dropout(0.3),
        Bidirectional(LSTM(50, return_sequences=False)),
        Dropout(0.3),
        Dense(50, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint(MODEL_FILE, save_best_only=True)

    model.fit(X, y, validation_split=0.2, epochs=50, batch_size=32, verbose=1, callbacks=[early_stopping, checkpoint])
    model.save(MODEL_FILE)

    with open(SCALER_FILE, 'wb') as f:
        pickle.dump(scaler, f)

    return model, scaler


# Load Model and Scaler
def load_model_and_scaler():
    model, scaler = None, None
    if os.path.exists(MODEL_FILE):
        model = load_model(MODEL_FILE)
    if os.path.exists(SCALER_FILE):
        with open(SCALER_FILE, 'rb') as f:
            scaler = pickle.load(f)
    return model, scaler


# Predict future expenses
def predict_expenses(model, scaler, df, sequence_length=30, days=30):
    last_sequence = df[['Amount', 'DayOfWeek', 'Month', 'Day']].values[-sequence_length:].reshape(1, sequence_length, 4)
    predictions = []

    for i in range(days):
        next_value = model.predict(last_sequence)[0][0]
        predictions.append(next_value)
        print(df.index[-1])
        next_features = np.array([[next_value, (df.index[-1] + timedelta(days=i + 1)).dayofweek,
                                   (df.index[-1] + timedelta(days=i + 1)).month,
                                   (df.index[-1] + timedelta(days=i + 1)).day]])
        last_sequence = np.roll(last_sequence, -1, axis=1)
        last_sequence[0, -1, :] = next_features

    predictions = np.array(predictions).reshape(-1, 1)
    return scaler.inverse_transform(predictions)


# Streamlit App
def main():
    st.title("Upcoming Expense Predictor")

    uploaded_file = st.file_uploader("Upload your expense data (CSV)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file, header=None)

        # Check the columns of the DataFrame
        st.write("Columns in the CSV:", df.columns)  # Print column names
        st.write("First few rows of the dataset:", df.head())  # Show first few rows to inspect the data

        # Define columns explicitly if needed, for example if the file doesn't have headers
        df.columns = ['Date', 'Amount', 'Description']
        df.columns = df.columns.str.strip()
        df = df[['Date', 'Amount']]
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
        st.dataframe(df.head())

        model, scaler = load_model_and_scaler()
        if model is None or scaler is None:
            if st.button("Train Model"):
                model, scaler = train_model(df)

        if model and scaler:
            st.subheader("Predicted Expenses for the Next 30 Days")
            forecast = predict_expenses(model, scaler, df)
            forecast_dates = pd.date_range(start=df['Date'].iloc[-1] + timedelta(days=1), periods=30)
            forecast_df = pd.DataFrame({'Date': forecast_dates, 'Predicted Amount': forecast.flatten()})
            st.dataframe(forecast_df)

            # Plot Predictions
            fig, ax = plt.subplots()
            ax.plot(forecast_df['Date'], forecast_df['Predicted Amount'], label='Predicted', linestyle='dashed',
                    color='red')
            ax.set_xlabel("Date")
            ax.set_ylabel("Amount")
            ax.set_title("Expense Prediction for Next 30 Days")
            ax.legend()
            st.pyplot(fig)


if __name__ == "__main__":
    main()
