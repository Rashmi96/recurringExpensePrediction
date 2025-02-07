import streamlit as st
import pandas as pd
from prophet import Prophet
import joblib
import os
from datetime import datetime, timedelta
from collections import Counter

# Configuration (using environment variables or defaults)
DATA_FILE = os.environ.get("DATA_FILE", "data/bank_statement.csv")
MODEL_FILE = os.environ.get("MODEL_FILE", "models/expense_model.joblib")

def load_data(filename):
    try:
        print(filename)
        df = pd.read_csv(filename, parse_dates=['Date'],header=None, names=['Date', 'Amount', 'Description'])
        print(df.head())
        if not {'Date', 'Amount', 'Description'}.issubset(df.columns):
            raise ValueError("CSV must contain 'Date', 'Amount', and 'Description' columns.")
        df['Amount'] = df['Amount'].abs() #Make all amount positive
        df = df[df['Amount']!=0] #Remove zero amount transactions
        df = df.sort_values('Date')
        df.drop_duplicates(subset=['Date', 'Amount', 'Description'], keep='first', inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.fillna(method='ffill', inplace=True)
        return df
    except FileNotFoundError:
        st.error(f"Data file not found: {filename}")
        return None
    except pd.errors.ParserError:
        st.error(f"Error parsing CSV file: {filename}. Check file format.")
        return None
    except ValueError as e:
        st.error(str(e))
        return None

def feature_engineering(df):
    df = df[['Date', 'Amount']]
    df = df.rename(columns={'Date': 'ds', 'Amount': 'y'})
    return df

def train_model(df):
    model = Prophet()
    model.fit(df)
    joblib.dump(model, MODEL_FILE)
    st.success("Model trained and saved!")
    return model

def load_model():
    try:
        model = joblib.load(MODEL_FILE)
        return model
    except FileNotFoundError:
        return None

def predict_expenses(model, days=7):
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    return forecast

def categorize_transaction(description):
    # Basic categorization (improve with more sophisticated logic or ML)
    description = description.lower()
    if any(keyword in description for keyword in ['grocery', 'food', 'restaurant']):
        return "Food"
    elif any(keyword in description for keyword in ['transport', 'bus', 'train', 'fuel', 'petrol']):
        return "Transportation"
    elif any(keyword in description for keyword in ['bill', 'utility', 'phone', 'internet']):
        return "Bills"
    elif any(keyword in description for keyword in ['amazon', 'flipkart', 'online']):
        return "Shopping"
    elif any(keyword in description for keyword in ['movie', 'concert', 'game']):
        return "Entertainment"
    else:
        return "Other"

def find_recurring_payments(df, min_occurrences=2):
    """Finds recurring payments based on description and frequency."""
    df['Category'] = df['Description'].apply(categorize_transaction)
    recurring_payments = {}

    for category in df['Category'].unique():
        category_df = df[df['Category'] == category]
        description_counts = Counter(category_df['Description'])

        for description, count in description_counts.items():
            if count >= min_occurrences:
                dates = category_df[category_df['Description'] == description]['Date'].dt.date.tolist()
                date_diffs = [(dates[i+1] - dates[i]).days for i in range(len(dates) - 1)]
                if date_diffs:
                    most_common_diff = Counter(date_diffs).most_common(1)[0][0]
                    recurring_payments[description] = {
                        'category': category,
                        'frequency': most_common_diff,
                        'next_due': dates[-1] + timedelta(days=most_common_diff)
                    }

    return recurring_payments

def generate_recurring_notifications(recurring_payments):
    notifications = []
    today = datetime.now().date()
    for description, details in recurring_payments.items():
        if details['next_due'] <= today + timedelta(days=7) and details['next_due'] >= today:
            notifications.append({
                "description": description,
                "category": details['category'],
                "due_date": details['next_due'].strftime("%Y-%m-%d"),
                "message": f"Recurring payment for {description} (Category: {details['category']}) is due on {details['next_due'].strftime('%Y-%m-%d')}."
            })
    return notifications

def main():
    st.title("Bank Statement Expense Prediction and Notification App")

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
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                model = train_model(df_prophet)
        else:
            model = load_model()
            if model is None:
                st.warning("No trained model found. Please train a model first.")

        if model:
            recurring_payments = find_recurring_payments(df.copy())
            if recurring_payments:
                st.subheader("Recurring Payments:")
                for desc, details in recurring_payments.items():
                    st.write(f"- {desc}: Category: {details['category']}, Frequency: {details['frequency']} days, Next Due: {details['next_due'].strftime('%Y-%m-%d')}")

                recurring_notifications = generate_recurring_notifications(recurring_payments)
                if recurring_notifications:
                    st.subheader("Recurring Payment Notifications (Next 7 Days):")
                    for notification in recurring_notifications:
                        st.warning(notification['message'])
                else:
                    st.info("No recurring payments due in the next 7 days.")
            else:
                st.info("No recurring payments found.")

if __name__ == "__main__":
    main()