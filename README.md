# Bank Statement Recurring Expense Detector

## Overview
The **Bank Statement Recurring Expense Detector** is a Streamlit-based web application designed to analyze bank statements from PDFs and CSV files. The application extracts transaction data, identifies recurring expenses, predicts future expenses using machine learning techniques, and visualizes expense trends.

## Features
- Extract transactions from PDF and CSV bank statements.
- Identify recurring expenses based on historical transaction patterns.
- Predict future expenses using **Linear Regression**.
- Visualize transaction trends and future projections with interactive plots.
- Notify users about projected upcoming expenses to help in financial planning.

## High-Level Design

### Architecture Overview

```plaintext
                    +-------------------------+
                    |      User Uploads       |
                    |  (PDF / CSV Statement)  |
                    +-----------+-------------+
                                |
                  +-------------v------------+
                  |   File Processing Layer  |
                  | - Extract Text (PDF)     |
                  | - Parse Transactions     |
                  | - Read CSV               |
                  +-------------+------------+
                                |
                  +-------------v------------+
                  |   Data Processing Layer  |
                  | - Identify Recurring Tx  |
                  | - Train ML Model         |
                  | - Predict Future Expense |
                  +-------------+------------+
                                |
                  +-------------v------------+
                  |       Visualization      |
                  | - Display Transactions   |
                  | - Plot Trend Graphs      |
                  | - Notify User            |
                  +-------------------------+
```

## Installation
### Prerequisites
- Python 3.9+
- pip (Python package manager)

### Creating a Virtual Environment
To ensure a clean and isolated environment, create a virtual environment before installing dependencies:
```sh
python -m venv bank_env
python -m venv bank_env # On macOS/Linux
bank_env\Scripts\activate     # On Windows
```

### Install Dependencies
```sh
pip install -r requirements.txt
```

## Usage
### Running the Application
```sh
streamlit run app.py
```

### Uploading Files
1. Open the web application in your browser.
2. Upload a **PDF** or **CSV** file containing transaction history.
3. View extracted transactions, recurring expenses, and predictions.

### Expected File Formats
#### CSV Format:
```
date,amount,description
2024-01-01,-100.50,Gym Membership
2024-02-01,-100.50,Gym Membership
```
#### PDF Format:
- Must contain transaction details with **date**, **amount**, and **description** in readable text.

## Key Components
### 1. File Processing
- **PDF Extraction**: Uses `pdfplumber` to extract readable text from PDFs.
- **CSV Parsing**: Reads CSV files, ensuring required columns (`date`, `amount`, `description`) are present.

### 2. Transaction Analysis
- Extracts transaction details using **regular expressions**.
- Identifies recurring expenses by analyzing transaction frequency and patterns.

### 3. Prediction Model
- Uses **Linear Regression** (from `sklearn.linear_model`) to predict future expenses.
- Projects next month's expected expenses based on past data trends.
- Identifies financial patterns to help users plan their budgets efficiently.

### 4. Model Selection
- The **Linear Regression** model was chosen for its simplicity and effectiveness in capturing trends over time.
- It is well-suited for financial forecasting when dealing with historical data.
- Future versions may explore **time series models** like ARIMA or LSTMs for improved accuracy.

### 5. Data Visualization
- Uses `matplotlib` and `seaborn` to generate interactive trend graphs.
- Highlights predicted expenses and past spending patterns.
- Provides a clear representation of financial trends over time.

### 6. Notifications & Alerts
- Displays projected expenses and advises users to maintain a sufficient balance.
- Sends reminders based on predicted expenses to help users avoid financial shortfalls.

## Example Output
- **Extracted Transactions Table**
- **Recurring Expense Summary**
- **Predicted Expenses Table**
- **Expense Trend Graph**

## Future Enhancements
- Support for multiple currencies and localized date formats.
- Advanced ML models for improved forecasting accuracy.
- Integration with financial APIs for real-time transaction tracking.
- User authentication and secure storage of past reports.

## License
This project is licensed under the MIT License. Feel free to use and modify it as needed.

