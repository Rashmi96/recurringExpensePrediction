import streamlit as st
import pandas as pd
import pdfplumber
from dateutil.parser import parse
import re
from io import BytesIO
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates


# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    return text


# Function to parse transactions from text
def parse_transactions(text):
    pattern = r"(\d{2,4}[-/]\d{2}[-/]\d{2})\s+([-]?\d+[.,]?\d*)\s+([\w\s]+)"
    matches = re.findall(pattern, text)
    transactions = []

    for m in matches:
        try:
            date = parse(m[0]).date()
            amount = float(m[1].replace(',', ''))
            description = m[2].strip()
            transactions.append({"date": date, "amount": amount, "description": description})
        except Exception as e:
            continue

    return pd.DataFrame(transactions)


# Function to parse CSV
def parse_csv(file):
    df = pd.read_csv(file)
    df.rename(columns=lambda x: x.strip().lower(), inplace=True)
    required_columns = {"date", "amount", "description"}
    if not required_columns.issubset(df.columns):
        st.error("CSV file must contain 'date', 'amount', and 'description' columns.")
        return pd.DataFrame()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    return df


# Function to identify recurring expenses
def identify_recurring_expenses(df, tolerance=3):
    df['date'] = pd.to_datetime(df['date'])
    recurring_expenses = {}

    for desc in df['description'].unique():
        data = df[df['description'] == desc].sort_values(by='date')
        intervals = data['date'].diff().dt.days.dropna()
        if len(intervals) > 1 and np.all(np.abs(intervals - intervals.median()) <= tolerance):
            recurring_expenses[desc] = data

    return recurring_expenses


# Function to fit trend and predict next month's expense
def fit_trend_and_predict(data):
    data['days_since_first'] = (data['date'] - data['date'].min()).dt.days
    model = LinearRegression()
    model.fit(data[['days_since_first']], data['amount'])

    next_day = (data['date'].max() + pd.Timedelta(days=30) - data['date'].min()).days
    predicted_amount = model.predict([[next_day]])[0]

    return predicted_amount, next_day


# Function to predict next month's expenses
def predict_next_month_expenses(recurring_expenses):
    predictions = []
    for desc, data in recurring_expenses.items():
        latest_data = data.iloc[-1:]
        predicted_amount, next_day = fit_trend_and_predict(data)

        predictions.append({
            "description": desc,
            "date": latest_data['date'].max() + pd.Timedelta(days=30),
            "amount": predicted_amount,
            "predicted_day": next_day
        })

    return pd.DataFrame(predictions)


# Function to plot the expenses with trend and predictions
def plot_expenses_with_trend(data, predicted_data):
    sns.set(style="whitegrid", palette="muted")

    plt.figure(figsize=(12, 6))

    # Plot historical expenses as a line plot
    plt.plot(data['date'], data['amount'], color='dodgerblue', marker='o', markersize=6, linestyle='-',
             label='Historical Expenses')

    # Smooth trend line using LinearRegression
    data['days_since_first'] = (data['date'] - data['date'].min()).dt.days
    model = LinearRegression()
    model.fit(data[['days_since_first']], data['amount'])
    trend_line = model.predict(data[['days_since_first']])
    plt.plot(data['date'], trend_line, color='seagreen', linestyle='--', linewidth=2, label='Trend Line')

    # Plot predicted expenses
    plt.scatter(predicted_data['date'], predicted_data['amount'], color='tomato', marker='x', s=100,
                label='Predicted Expenses')

    # Customize the plot with title, labels, and formatting
    plt.title("Recurring Expenses with Trend and Predictions", fontsize=16, fontweight='bold')
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Amount ($)", fontsize=14)

    # Rotate date labels for better readability
    plt.xticks(rotation=45)

    # Format the x-axis with date
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

    # Add a legend
    plt.legend(fontsize=12)

    # Show the plot
    st.pyplot(plt)


# Function to notify the user to maintain sufficient balance
def notify_user_to_maintain_balance(predicted_expenses):
    total_predicted = predicted_expenses['amount'].sum()
    message = f"Your projected expenses for the next month are ${total_predicted:.2f}. Please ensure that you have sufficient balance in your savings account to cover these expenses."
    st.info(message)


# Main function to run the Streamlit app
def main():
    st.title("Bank Statement Recurring Expense Detector")

    uploaded_file = st.file_uploader("Upload your bank statement (PDF or CSV)", type=["pdf", "csv"])

    if uploaded_file:
        if uploaded_file.name.endswith(".pdf"):
            text = extract_text_from_pdf(BytesIO(uploaded_file.read()))
            transactions_df = parse_transactions(text)
        else:
            transactions_df = parse_csv(uploaded_file)

        if not transactions_df.empty:
            st.subheader("Extracted Transactions")
            st.dataframe(transactions_df)

            recurring_expenses = identify_recurring_expenses(transactions_df)

            st.subheader("Latest Recurring Expenses")
            if recurring_expenses:
                latest_expenses = []

                for desc, data in recurring_expenses.items():
                    latest_value = data.iloc[-1]
                    latest_expenses.append({
                        "description": desc,
                        "date": latest_value['date'].strftime('%Y-%m-%d'),
                        "amount": f"${latest_value['amount']:.2f}"
                    })

                latest_expenses_df = pd.DataFrame(latest_expenses)
                st.dataframe(latest_expenses_df)

                predictions_df = predict_next_month_expenses(recurring_expenses)
                st.subheader("Predicted Expenses for Next Month")
                st.dataframe(predictions_df)

                # Plot the historical and predicted expenses with the trend line
                all_data = pd.concat([transactions_df, predictions_df[['date', 'amount']]])
                all_data['amount'] = all_data['amount'].astype(float)

                # Plot expenses
                plot_expenses_with_trend(transactions_df, predictions_df)

                # Notify the user to maintain sufficient balance
                notify_user_to_maintain_balance(predictions_df)
            else:
                st.write("No recurring expenses detected.")


if __name__ == "__main__":
    main()
