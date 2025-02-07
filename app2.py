from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib
import streamlit as st
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import streamlit as st

# Configuration
DATA_FILE = os.environ.get("DATA_FILE", "data/expense.csv")
MODEL_FILE = os.environ.get("MODEL_FILE", "models/lstm_model.h5")
SCALER_FILE = os.environ.get("SCALER_FILE", "models/scaler.pkl")

# Function to preprocess the data for LSTM
def preprocess_data(df, sequence_length=60):
    data = df[['Date', 'Amount']].dropna()  # Assuming 'Amount' is the feature you're predicting
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Scale the 'Amount' column
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Amount']])

    # Prepare data in sequences of 'sequence_length'
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i - sequence_length:i, 0])  # Features (last 'sequence_length' values)
        y.append(scaled_data[i, 0])  # Target (the next value after the sequence)

    # Convert to numpy arrays
    X, y = np.array(X), np.array(y)

    # Reshape X to be 3D for LSTM (samples, timesteps, features)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler

import os
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler

def train_model(df, sequence_length=60, model_dir="models"):
    try:
        # Ensure the required columns exist
        required_columns = ['Date', 'Amount']
        if not all(col in df.columns for col in required_columns):
            raise KeyError(f"Missing required columns. Expected {required_columns}, but got {df.columns.tolist()}")

        # Convert Date column to datetime and sort data
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')

        # Scale the Amount column
        scaler = MinMaxScaler(feature_range=(0, 1))
        df['Amount'] = scaler.fit_transform(df[['Amount']])

        # Create sequences for LSTM
        X, y = [], []
        for i in range(sequence_length, len(df)):
            X.append(df['Amount'].iloc[i-sequence_length:i].values)
            y.append(df['Amount'].iloc[i])

        X, y = np.array(X), np.array(y)

        # Ensure data is correctly shaped
        if X.shape[0] == 0 or y.shape[0] == 0:
            raise ValueError("Training data is empty. Ensure there is enough historical data.")

        # Reshape X for LSTM input (samples, time steps, features)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # Build LSTM model
        model = Sequential([
            Input(shape=(X.shape[1], 1)),  # Input layer to avoid warning
            LSTM(units=50, return_sequences=True),
            LSTM(units=50, return_sequences=False),
            Dense(units=25),
            Dense(units=1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')

        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_path = os.path.join(model_dir, "lstm_model.h5")
        scaler_path = os.path.join(model_dir, "scaler.pkl")

        # Callbacks for training
        early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        checkpoint = ModelCheckpoint(model_path, monitor='loss', save_best_only=True, verbose=1)

        # Train the model
        model.fit(X, y, epochs=20, batch_size=32, verbose=1, callbacks=[early_stopping, checkpoint])

        # Explicitly save the final trained model
        model.save(model_path)
        print(f"Model saved at: {model_path}")

        # Save the scaler using pickle
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved at: {scaler_path}")

        return model, scaler

    except Exception as e:
        print(f"Error in training model: {str(e)}")
        return None, None



# Load Model and Scaler Function
def load_model_and_scaler():
    model, scaler = None, None
    if os.path.exists(MODEL_FILE):
        model = keras_load_model(MODEL_FILE, compile=False)  # Load model without compilation
        model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())

    if os.path.exists(SCALER_FILE):
        scaler = joblib.load(SCALER_FILE)  # Load scaler if it exists

    return model, scaler


def load_data(filename):
    try:
        # Try reading the CSV file without header and infer columns
        df = pd.read_csv(filename, header=None)

        # Check the columns of the DataFrame
        st.write("Columns in the CSV:", df.columns)  # Print column names
        st.write("First few rows of the dataset:", df.head())  # Show first few rows to inspect the data

        # Define columns explicitly if needed, for example if the file doesn't have headers
        df.columns = ['Date', 'Amount', 'Description']  # Assuming these columns exist, but we'll validate

        # Strip any leading or trailing spaces from column names
        df.columns = df.columns.str.strip()

        # Verify if expected columns are present
        if not all(col in df.columns for col in ['Date', 'Amount', 'Description']):
            raise ValueError("CSV must contain 'Date', 'Amount', and 'Description' columns.")

        df['Amount'] = df['Amount'].abs()
        df = df[df['Amount'] != 0]
        df = df.sort_values('Date')
        df.drop_duplicates(subset=['Date', 'Amount', 'Description'], keep='first', inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.ffill(inplace=True)

        # Ensure 'Date' is in datetime format
        df['Date'] = pd.to_datetime(df['Date'])
        df['ds'] = df['Date']  # Create 'ds' column for future use

        # Add cyclical features
        df['day_of_week'] = df['ds'].dt.dayofweek
        df['month'] = df['ds'].dt.month
        df['day_of_year'] = df['ds'].dt.dayofyear

        # One-hot encode 'Description' column
        if 'Description' in df.columns:
            df = pd.get_dummies(df, columns=['Description'], drop_first=True)
        else:
            st.warning("No 'Description' column found in the dataset.")

        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def feature_engineering(df):
    try:
        # Check if 'Amount' or the target column is present to create 'y'
        if 'Amount' not in df.columns:
            st.error("'Amount' column is missing in the data!")
            return None

        # Set 'y' as 'Amount' for the prediction task
        df['y'] = df['Amount']  # Ensure this step is done correctly

        # Continue with the rest of the feature engineering...

        return df
    except Exception as e:
        st.error(f"Error in feature engineering: {str(e)}")
        return None


# Prediction Function
def predict_expenses(model, scaler, data, sequence_length=5, days=30):
    last_sequence = data['y'].values[-sequence_length:].reshape(1, sequence_length, 1)  # Reshape for LSTM input
    predictions = []

    for _ in range(days):
        next_value = model.predict(last_sequence)
        predictions.append(next_value[0][0])  # Store the predicted value
        last_sequence = np.roll(last_sequence, shift=-1, axis=1)
        last_sequence[0, -1, 0] = next_value[0][0]

    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)
    return predictions

# Notification Generation Function
def generate_notifications(df, forecast, threshold_percentage=20):
    notifications = []
    today = datetime.now()
    for index, row in forecast.iterrows():
        if row['ds'] > today and row['ds'] <= today + timedelta(days=7):
            predicted_expense = row['yhat']
            date_str = row['ds'].strftime("%Y-%m-%d")

            # Compare with last expense or average expenses for this day of the week
            past_expenses_same_day_of_week = df[df['ds'].dt.dayofweek == row['ds'].dayofweek]['y'].values
            if past_expenses_same_day_of_week.size > 0:
                average_past_expense = past_expenses_same_day_of_week.mean()
                if predicted_expense > (average_past_expense * (1 + threshold_percentage / 100)):
                    notifications.append({
                        "date": date_str,
                        "expense": predicted_expense,
                        "message": f"Your predicted expense on {date_str} is ₹{predicted_expense:.2f}, which is more than {threshold_percentage}% higher than your average past expense for this day of the week."
                    })
                elif predicted_expense < (average_past_expense * (1 - threshold_percentage / 100)):
                    notifications.append({
                        "date": date_str,
                        "expense": predicted_expense,
                        "message": f"Your predicted expense on {date_str} is ₹{predicted_expense:.2f}, which is more than {threshold_percentage}% lower than your average past expense for this day of the week."
                    })
            elif predicted_expense > (df['y'].iloc[-1] * (1 + threshold_percentage / 100)):
                notifications.append({
                    "date": date_str,
                    "expense": predicted_expense,
                    "message": f"Your predicted expense on {date_str} is ₹{predicted_expense:.2f}, which is more than {threshold_percentage}% higher than your last expense."
                })
            elif predicted_expense < (df['y'].iloc[-1] * (1 - threshold_percentage / 100)):
                notifications.append({
                    "date": date_str,
                    "expense": predicted_expense,
                    "message": f"Your predicted expense on {date_str} is ₹{predicted_expense:.2f}, which is more than {threshold_percentage}% lower than your last expense."
                })
    return notifications

# Main Function
def main():
    st.title("Bank Statement Expense Prediction and Notification App")

    # Data loading
    uploaded_file = st.file_uploader("Upload your bank statement (CSV)", type="csv")
    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)
            if df is not None:
                st.write("Data loaded successfully:")
                st.dataframe(df.head())
        except ValueError as e:
            st.error(str(e))
            df = None
    else:
        st.info("Please upload a bank statement.")
        return

    if df is not None:
        df_prophet = feature_engineering(df.copy())

        # Model training/loading
        model, scaler = load_model_and_scaler()
        if model is None or scaler is None:
            if st.button("Train Model"):
                with st.spinner("Training model..."):
                    model, scaler = train_model(df_prophet)
            else:
                st.warning("No trained model found. Please train a model first.")

        if model:
            st.subheader("Upcoming Expenses and Notifications")
            forecast = predict_expenses(model, scaler, df_prophet, sequence_length=5, days=30)
            st.write("Forecasted Expenses for the next 7 days:")
            forecast_df = pd.DataFrame({'ds': pd.date_range(start=df_prophet['ds'].iloc[-1], periods=30),
                                       'yhat': forecast.flatten()})
            st.dataframe(forecast_df)

            notifications = generate_notifications(df_prophet, forecast_df)
            if notifications:
                st.subheader("Expense Notifications:")
                for notification in notifications:
                    st.warning(notification['message'])
            else:
                st.info("No significant expense changes predicted for the next 7 days.")

if __name__ == "__main__":
    main()
