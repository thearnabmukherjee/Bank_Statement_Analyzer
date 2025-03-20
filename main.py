import streamlit as st
import os
import tempfile
import logging
import json
import re
import asyncio
import nest_asyncio
from datetime import datetime, timedelta

import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from PIL import Image
from pypdf import PdfReader
import io
import msoffcrypto
import pytesseract
import docx2txt
import pandas as pd
import zipfile
from pathlib import Path
import numpy as np

# Apply nest_asyncio
nest_asyncio.apply()

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

# Set up Tesseract OCR path for Windows
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Load LLM model
gemini_api_key = os.getenv('GEMINI_API_KEY')
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=gemini_api_key, temperature=0)

financial_prompt = ChatPromptTemplate.from_template("""
    You are analyzing a bank statement and structuring each transaction into a strict JSON format. 

    **For each transaction, ensure the following rules are met:**

    - **Valid JSON Structure:** No missing delimiters or incomplete expressions.
    - **JSON Transaction format:**
    
    {{
        "transactions":[{{
            "date": "YYYY-MM-DD",
            "amount": 0.0,
            "balance_remaining": 0.0,
            "transaction_type": "str",
            "description": "",
            "category": "str",
            "is_weekend": boolean
        }}]
    }}

    **Strict Data Rules:**
    - "date": Must be in YYYY-MM-DD format. [**STRICT!**] Infer the date format from the document **BY OBSERVING MULTIPLE DATES** and convert all the dates accordingly.
    - "amount": Absolute numerical accuracy without commas (e.g., 7500.00, not 7,500.00). Must be greater than zero.
    - "balance_remaining": Extract the balance after the transaction accurately from the document.
    - "transaction_type": Either "Credit" or "Debit".
    - "description": Give one or two word description of the transaction, generally would be present in the document.
    - "category": 
        - Categorize based on predefined categories (e.g., Travel, Groceries, Food&Dining, Shopping, Rent, Medical, Utilities, Movies&Entertainment, Fuel&Transportation, Credit Card Payment, Investment, etc.).
        - Do not leave this field empty or unknown! Assign a category using past classification patterns. 
        - Try as hard as you can to categorise it among the predefined categories, unless there is no possible category, assign it Others.
    - "is_weekend": true if the date falls on a **Saturday or Sunday**, else false.

    **Strict Compliance Guidelines:**
    - Do not merge, summarize, or remove transactions.
    - Ensure all transactions are **fully structured and accurate** before closing the JSON.
    - Avoid raw calculations or unresolved expressions—all values must be precomputed.

    **Input:**  
    - A list of raw transactions with details such as date, amount, balance, description, and type.

    **Output:**  
    - A valid JSON object for each transaction in the format specified. Only a JSON object with no unnecessary extra characters.

    **Process all transactions accurately from the document and return a structured JSON output.**

<context>
{context}
<context>
""")

analysis_prompt = ChatPromptTemplate.from_template("""
    You are a financial advisor assessing bank transactions for detailed insights, risk indicators, and recommendations. 
    Use the provided JSON data to generate a comprehensive analysis. Avoid mentioning any currency in the analysis.

    # Financial Behavior Analysis
    1. Identify income sources (e.g., transactions categorized as "Salary").
    2. Classify transactions:
    - **Essential**: Grocery, Utilities, Travel (commuting-related).
    - **Discretionary**: Shopping, Movies&Entertainment, Food&Dining (non-essential).
    3. Detect recurring savings or debt payments (e.g., Investment, Credit Card Payment) or recurring transactions (credit or debit) in general.

    # Risk Indicators
    - Flag high-risk spending (e.g., gambling, smoking, drinking—keywords: "casino", "bar", "liquor", "tobacco") impacting financial stability.
    - Assess discretionary spending vs. income (e.g., excessive discretionary spending exceeding 50 percent of inflow).
    - Flag a transaction as suspicious if the amount is unusually large or if there are multiple transactions with the same amount within a short period.

    # Loan Eligibility
    - **Eligible**: Stable income, discretionary spending <30 percent of inflow, no risks.
    - **Partially Eligible**: Moderate discretionary spending (30-50%), manageable expenses.
    - **Not Eligible**: High discretionary spending (>50%), unstable income, or risk indicators.

    # Financial Health Recommendations
    - **General**: Reduce discretionary spending, increase savings.
    - **Weekdays**: Cut incidental spending (e.g., frequent small Travel/Food&Dining expenses).
    - **Weekends**: Limit entertainment and dining out (e.g., Movies&Entertainment, Food&Dining).
    - Provide tips on how the user can save more and/or increase their credit score.

    # Suspicious and Recurring Transactions
    - Flag a transaction as recurring if it occurs regularly (e.g., monthly) with the same amount and category.
    - Provide a summary of suspicious and recurring transactions along with any patterns or insights you can derive.

    **Analyze all transactions accurately from the document and provide a readable summary.**

    <context>
    {context}
    <context>
""")

# Enhanced AML Risk Score Calculation
def calculate_aml_risk_score(df_transactions):
    """
    Calculate AML Risk Score based on transaction data with enhanced factors.
    Score ranges from 0 to 100, where higher scores indicate higher risk.
    
    Parameters:
    - df_transactions (pd.DataFrame): DataFrame containing transaction data
    
    Returns:
    - int: AML Risk Score (0-100)
    """
    if df_transactions.empty:
        return 0  # No transactions, no risk

    # Initialize weights for each factor (total weight should sum to 1.0)
    weights = {
        'high_amount': 0.20,           # High transaction amounts
        'frequent_transactions': 0.15, # Frequent transactions in a short period
        'high_risk_categories': 0.20,  # High-risk categories
        'weekend_transactions': 0.10,  # Weekend transactions
        'balance_fluctuations': 0.15,  # Balance fluctuations
        'rapid_inflow_outflow': 0.15,  # Rapid inflows followed by outflows
        'cash_transactions': 0.10,     # Cash-related transactions
        'anomaly_detection': 0.10,     # Anomalies in transaction amounts
        'recurring_suspicious': 0.10,  # Recurring transactions in high-risk categories
        'activity_variance': 0.10      # Variance in transaction frequency
    }

    # Initialize scores for each factor (0-100 scale)
    factor_scores = {key: 0 for key in weights.keys()}

    # 1. High transaction amounts (contributes up to 100 points for this factor)
    amount_threshold = df_transactions['amount'].quantile(0.90)
    high_amount_transactions = df_transactions[df_transactions['amount'] > amount_threshold]
    if not high_amount_transactions.empty:
        factor_scores['high_amount'] = min(100, len(high_amount_transactions) * 20)  # 20 points per high transaction

    # 2. Frequent transactions in a short period (contributes up to 100 points for this factor)
    df_transactions = df_transactions.sort_values(by='date')
    df_transactions['date_diff'] = df_transactions['date'].diff().dt.days
    df_transactions['same_amount'] = df_transactions['amount'].eq(df_transactions['amount'].shift())
    frequent_transactions = df_transactions[(df_transactions['date_diff'] <= 1) & (df_transactions['same_amount'])]
    if not frequent_transactions.empty:
        factor_scores['frequent_transactions'] = min(100, len(frequent_transactions) * 20)  # 20 points per occurrence

    # 3. High-risk categories (contributes up to 100 points for this factor)
    high_risk_categories = ['casino', 'bet', 'gambling', 'luxury', 'jewelry', 'nightclub', 'bar', 'liquor', 'tobacco']
    high_risk_transactions = df_transactions[df_transactions['category'].str.lower().str.contains('|'.join(high_risk_categories), na=False)]
    if not high_risk_transactions.empty:
        factor_scores['high_risk_categories'] = min(100, len(high_risk_transactions) * 20)  # 20 points per high-risk transaction

    # 4. Weekend transactions (contributes up to 100 points for this factor)
    weekend_transactions = df_transactions[df_transactions['is_weekend']]
    if not weekend_transactions.empty:
        factor_scores['weekend_transactions'] = min(100, len(weekend_transactions) * 10)  # 10 points per weekend transaction

    # 5. Balance fluctuations (contributes up to 100 points for this factor)
    max_balance = df_transactions['balance_remaining'].max()
    min_balance = df_transactions['balance_remaining'].min()
    if max_balance > 0:  # Avoid division by zero
        balance_fluctuation_ratio = (max_balance - min_balance) / max_balance
        factor_scores['balance_fluctuations'] = int(balance_fluctuation_ratio * 100)  # Scale to 0-100

    # 6. Rapid inflows followed by outflows (contributes up to 100 points for this factor)
    rapid_inflow_outflow = 0
    for i in range(len(df_transactions) - 1):
        if (df_transactions.iloc[i]['transaction_type'].lower() == 'credit' and 
            df_transactions.iloc[i + 1]['transaction_type'].lower() == 'debit'):
            time_diff = (df_transactions.iloc[i + 1]['date'] - df_transactions.iloc[i]['date']).days
            if time_diff <= 3:  # Within 3 days
                rapid_inflow_outflow += 1
    if rapid_inflow_outflow > 0:
        factor_scores['rapid_inflow_outflow'] = min(100, rapid_inflow_outflow * 25)  # 25 points per occurrence

    # 7. Cash transactions (contributes up to 100 points for this factor)
    cash_keywords = ['atm', 'cash', 'withdrawal', 'deposit']
    cash_transactions = df_transactions[df_transactions['description'].str.lower().str.contains('|'.join(cash_keywords), na=False)]
    if not cash_transactions.empty:
        factor_scores['cash_transactions'] = min(100, len(cash_transactions) * 20)  # 20 points per cash transaction

    # 8. Anomaly detection (contributes up to 100 points for this factor)
    if not df_transactions.empty:
        mean_amount = df_transactions['amount'].mean()
        std_amount = df_transactions['amount'].std()
        if std_amount > 0:  # Avoid division by zero
            z_scores = np.abs((df_transactions['amount'] - mean_amount) / std_amount)
            anomalies = df_transactions[z_scores > 3]  # Transactions with z-score > 3 are considered anomalies
            if not anomalies.empty:
                factor_scores['anomaly_detection'] = min(100, len(anomalies) * 20)  # 20 points per anomaly

    # 9. Recurring transactions in high-risk categories (contributes up to 100 points for this factor)
    recurring_high_risk = df_transactions[df_transactions['flag'].str.contains('Recurring', na=False) & 
                                         df_transactions['category'].str.lower().str.contains('|'.join(high_risk_categories), na=False)]
    if not recurring_high_risk.empty:
        factor_scores['recurring_suspicious'] = min(100, len(recurring_high_risk) * 25)  # 25 points per recurring high-risk transaction

    # 10. Activity variance (contributes up to 100 points for this factor)
    if not df_transactions.empty:
        df_transactions['date'] = pd.to_datetime(df_transactions['date'])
        daily_transactions = df_transactions.groupby(df_transactions['date'].dt.date).size()
        activity_variance = daily_transactions.var() if len(daily_transactions) > 1 else 0
        if activity_variance > 0:
            # Normalize variance to a 0-100 scale (using a reasonable upper bound for variance)
            max_variance = 50  # Adjust based on your data
            factor_scores['activity_variance'] = min(100, (activity_variance / max_variance) * 100)

    # Calculate weighted final score
    final_score = 0
    for factor, score in factor_scores.items():
        final_score += score * weights.get(factor, 0)

    # Ensure the score is between 0 and 100
    final_score = min(100, max(0, int(final_score)))
    return final_score


def display_financial_summary_card(aml_risk_score, daily_avg_balance, max_balance, min_balance, days_gap, transactions, max_dormant_days, date_range):
    """
    Display a financial summary card with key metrics in two vertical columns with hover effects.
    
    Parameters:
    - aml_risk_score (int): AML risk score
    - daily_avg_balance (float): Daily average balance
    - max_balance (float): Maximum balance
    - min_balance (float): Minimum balance
    - days_gap (int): Days gap between max and min balance
    - transactions (int): Number of transactions
    - max_dormant_days (int): Maximum dormant days
    - date_range (str): Date range for the transactions
    """
    # Add a title for the financial summary
    st.markdown("### Financial Summary", unsafe_allow_html=True)

    # Inject custom CSS for styling and hover effects
    st.markdown("""
        <style>
        .card-container {
            display: flex;
            flex-direction: row; /* Arrange columns side by side */
            justify-content: space-between; /* Space columns evenly */
            gap: 2rem; /* Space between columns */
            margin: 1.5rem 0; /* Vertical margin for spacing */
            width: 100%; /* Ensure the container takes full width */
            box-sizing: border-box; /* Include padding in width calculation */
        }
        .card-column {
            display: flex;
            flex-direction: column; /* Stack cards vertically within each column */
            gap: 1rem; /* Space between cards in a column */
            flex: 1; /* Allow columns to grow equally */
            min-width: 15rem; /* Minimum width for each column */
        }
        .card {
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 1.25rem; /* Increased padding for larger text */
            background-color: #ffffff;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            text-align: center;
            display: flex;
            flex-direction: column;
            justify-content: center; /* Center content vertically */
            min-height: 6rem; /* Increased height to accommodate larger text */
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
            background-color: #f9fafb; /* Subtle background change on hover */
        }
        .metric-label {
            font-size: 1.1rem; /* Increased font size for labels */
            font-weight: 900; /* Bold labels */
            color: #37231f;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.75rem; /* Increased spacing below labels */
            line-height: 1.2; /* Improve readability */
        }
        .metric-value {
            font-size: 1.5rem; /* Increased font size for values */
            font-weight: 1100; /* Extra bold values */
            color: #1f2737;
            line-height: 1.2;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 2.5rem; /* Increased height for larger text */
        }
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .card-container {
                flex-direction: column; /* Stack columns vertically on smaller screens */
                align-items: center; /* Center columns on smaller screens */
            }
            .card-column {
                width: 100%; /* Full width on smaller screens */
                max-width: 20rem; /* Limit width for better readability */
            }
            .card {
                min-height: 5.5rem; /* Slightly smaller height for mobile */
            }
            .metric-label {
                font-size: 0.85rem; /* Slightly smaller font for mobile */
            }
            .metric-value {
                font-size: 1.4rem; /* Slightly smaller font for mobile */
            }
        }
        </style>
    """, unsafe_allow_html=True)

    # Start the container to hold the two columns
    st.markdown('<div class="card-container">', unsafe_allow_html=True)

    # First column: AML Risk Score, Daily Avg Balance, Max Balance, Min Balance
    st.markdown('<div class="card-column">', unsafe_allow_html=True)
    first_column_metrics = [
        ("AML Risk Score", aml_risk_score),
        ("Daily Avg Balance", f"₹{daily_avg_balance:,.2f}"),
        ("Max Balance", f"₹{max_balance:,.2f}"),
        ("Min Balance", f"₹{min_balance:,.2f}"),
    ]
    for label, value in first_column_metrics:
        st.markdown(f"""
            <div class="card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Second column: Days Gap B/W Max & Min, Transactions, Max Dormant Days
    st.markdown('<div class="card-column">', unsafe_allow_html=True)
    second_column_metrics = [
        ("Days Gap B/W Max & Min", f"{days_gap} days"),
        (f"Transactions<br>({date_range})", transactions),
        ("Max Dormant Days", max_dormant_days),
    ]
    for label, value in second_column_metrics:
        st.markdown(f"""
            <div class="card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Close the container
    st.markdown('</div>', unsafe_allow_html=True)


# Refine transaction tagging
def refine_tagging(transactions):
    category_keywords = {
        'Travel': ['irctc', 'uber', 'ola', 'travel', 'makemytrip', 'uts', 'railway', 'metro', 'bus', 'flight', 'ticket'],
        'Food&Dining': ['swiggy', 'zomato', 'restaurant', 'dining', 'food Afternoon', 'eatery', 'cafe', 'buffet', 'dominos', 'pizza', 'kfc', 'mcdonalds', 'burgerking', 'subway', 'pizzahut', 'starbucks', 'ccd', 'barista', 'chai', 'coffee'],
        'Grocery': ['grocery', 'blinkit', 'bigbasket', 'fresh', 'big bazaar', 'd mart', 'grofers', 'reliance fresh', 'spencers', 'nilgiris', 'heritage'],
        'Shopping': ['amazon', 'flipkart', 'paytm', 'shopping', 'myntra', 'snapdeal', 'pantaloons', 'zudio', 'reliance trends', 'westside', 'lifestyle', 'shoppers stop', 'central', 'max', 'uniqlo', 'h&m', 'zara', 'bata'],
        'Movies&Entertainment': ['netflix', 'bookmyshow', 'entertainment', 'hotstar', 'prime', 'zee5', 'sonyliv', 'disney', 'imagica'],
        'Utilities': ['electricity', 'water', 'internet', 'utility', 'rent', 'maintenance', 'society', 'housekeeping', 'cleaning', 'maid'],
        'Fuel&Transportation': ['petrol', 'fuel', 'diesel', 'cng', 'gas station', 'fuel station'],
        'Credit Card Payment': ['cred', 'credit card', 'payment', 'emi', 'loan', 'interest', 'charge', 'fee', 'fine', 'penalty'],
        'Taxes': ['tax', 'gst', 'tds'],
        'Investment': ['zerodha', 'investment', 'nps', 'mutual fund', 'shares', 'stock', 'stock market'],
    }

    for t in transactions:
        description = t.get('description', '').lower()
        amount = float(t.get('amount', 0)) if t.get('amount') else 0.0
        transaction_type = t.get('transaction_type', 'Debit').lower()
        tag = t.get('category', 'Other')

        # Handle credit transactions with explicit patterns
        if transaction_type == 'credit' and amount > 0:
            if re.search(r'\b(salary|payroll|wages|employer|paycheck)\b', description, re.IGNORECASE):
                tag = 'Salary'
            elif re.search(r'\b(refund|rev|return)\b', description, re.IGNORECASE):
                tag = 'Refund'
            else:
                # Try keyword-based category inference
                for category, keywords in category_keywords.items():
                    pattern = r'\b(?:' + '|'.join(map(re.escape, keywords)) + r')\b'
                    if re.search(pattern, description, re.IGNORECASE):
                        tag = category
                        break

        # Handle debit transactions with category mapping
        elif transaction_type == 'debit' and amount > 0:
            for category, keywords in category_keywords.items():
                pattern = r'\b(?:' + '|'.join(map(re.escape, keywords)) + r')\b'
                if re.search(pattern, description, re.IGNORECASE):
                    tag = category
                    break

        t['tag'] = tag
        t['transaction_type'] = transaction_type.capitalize()

    return transactions

def flag_recurring_transactions(transactions):
    recurring = {}
    start_date = datetime(2024, 10, 31)
    transactions_with_dates = []
    
    for i, t in enumerate(transactions):
        date_str = t.get('date', 'Unknown')
        if date_str == 'Unknown' or not re.match(r'\d{4}-\d{2}-\d{2}', date_str):
            inferred_date = start_date + timedelta(days=i * 5)
            t['date'] = inferred_date.strftime('%Y-%m-%d')
        transactions_with_dates.append((t, datetime.strptime(t['date'], '%Y-%m-%d')))
    
    for i, (t1, date1) in enumerate(transactions_with_dates):
        for t2, date2 in transactions_with_dates[i+1:]:
            delta = (date2 - date1).days
            if t1['amount'] == t2['amount'] and t1['transaction_type'] == t2['transaction_type']:
                key = (t1['amount'], t1['transaction_type'])
                if 6 <= delta <= 8:
                    recurring[key] = "Weekly Recurring"
                elif 28 <= delta <= 31:
                    recurring[key] = "Monthly Recurring"
    
    for t in transactions:
        key = (t['amount'], t['transaction_type'])
        t['flag'] = recurring.get(key, 'None')
    return transactions

# Flag weekend vs. weekday spending
def flag_weekend_vs_weekday(transactions):
    weekend_spend = 0.0
    weekday_spend = 0.0
    for t in transactions:
        date = datetime.strptime(t['date'], '%Y-%m-%d')
        is_weekend = date.weekday() >= 5
        t['is_weekend'] = is_weekend
        if t['transaction_type'].lower() == 'debit':
            if is_weekend:
                t['flag'] = "Weekend Spending" if t['flag'] == 'None' else f"{t['flag']}, Weekend Spending"
                weekend_spend += t['amount']
            else:
                t['flag'] = "Weekday Spending" if t['flag'] == 'None' else f"{t['flag']}, Weekday Spending"
                weekday_spend += t['amount']
    return transactions, weekend_spend, weekday_spend

# Flag high-risk transactions
def flag_high_risk_transactions(transactions):
    high_risk_keywords = ['casino', 'bet', 'gambling', 'luxury', 'jewelry', 'nightclub', 'bar', 'liquor', 'tobacco']
    for t in transactions:
        if any(keyword in t.get('category', '').lower() for keyword in high_risk_keywords):
            t['flag'] = f"{t['flag']}, High Risk" if t['flag'] != 'None' else "High Risk"
    return transactions

def parse_file(file, file_extension):
    docs = []
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name
    
    if file_extension == "pdf":
        reader = PdfReader(temp_file_path)
        if reader.is_encrypted:
            # Create a container for the password input
            password_container = st.empty()
            # Display password input in the container
            password = password_container.text_input("Enter PDF Password:", type="password")
            if password:
                try:
                    success = reader.decrypt(password)
                    if success:
                        st.session_state.pdf_password = password
                        st.success("PDF decrypted successfully!")
                        # Clear the password input field after successful decryption
                        password_container.empty()
                    else:
                        st.error("Incorrect password. Please try again.")
                        st.stop()
                except Exception as e:
                    st.error(f"Error decrypting PDF: {e}")
                    st.stop()
            else:
                st.warning("PDF is encrypted. Please enter the password.")
                st.stop()
        docs = [Document(page_content=page.extract_text()) for page in reader.pages]
    elif file_extension == "txt":
        docs = [Document(page_content=open(temp_file_path, "r", encoding="utf-8").read())]
    elif file_extension == "docx":
        docs = [Document(page_content=docx2txt.process(temp_file_path))]
    elif file_extension in ["jpg", "jpeg", "png"]:
        docs = [Document(page_content=pytesseract.image_to_string(Image.open(temp_file_path)))]
    elif file_extension == "xlsx":
        with open(temp_file_path, "rb") as f:
            office_file = msoffcrypto.OfficeFile(f)
            if office_file.is_encrypted():
                # Create a container for the password input
                password_container = st.empty()
                # Display password input in the container
                password = password_container.text_input("Enter Excel Password:", type="password")
                if password:
                    try:
                        decrypted_file = io.BytesIO()
                        office_file.load_key(password=password)
                        office_file.decrypt(decrypted_file)
                        df = pd.read_excel(decrypted_file)
                        docs = [Document(page_content=df.to_string())]
                        st.success("Excel file decrypted successfully!")
                        # Clear the password input field after successful decryption
                        password_container.empty()
                    except Exception as e:
                        st.error(f"Error decrypting Excel: {e}")
                        st.stop()
                else:
                    st.warning("Excel file is encrypted. Please enter the password.")
                    st.stop()
            else:
                # File is not encrypted, read directly
                df = pd.read_excel(temp_file_path)
                docs = [Document(page_content=df.to_string())]
    elif file_extension == "csv":
        try:
            df = pd.read_csv(temp_file_path)  # Directly read the CSV file
            docs = [Document(page_content=df.to_string())]
            st.success("CSV file processed successfully!")
        except Exception as e:
            st.error(f"Error processing CSV: {e}")
            st.stop()

    return docs

async def generate_json_from_llm(chunk_context, context, prompt):
    prompt = prompt.format(context=chunk_context)
    response = await llm.ainvoke(prompt)
    if not response.content:
        raise ValueError("The response content is empty.")
    
    cleaned_content = response.content.strip().strip("```json").strip("```")
    logging.info(f"Cleaned response content: {cleaned_content}")
    try:
        json_output = json.loads(cleaned_content)
    except json.JSONDecodeError as e:
        logging.error(f"JSONDecodeError: {e}")
        logging.info(f"Raw response content: {response.content}")
        raise ValueError("Failed to decode JSON from response content.")
    return json_output

async def generate_json_chunks(chunks, context, prompt):
    tasks = [generate_json_from_llm(" ".join(chunk), context, prompt) for chunk in chunks]
    json_chunks = await asyncio.gather(*tasks, return_exceptions=True)  # Batch processing with error handling
    combined_json = {"transactions": []}
    for chunk in json_chunks:
        if isinstance(chunk, dict) and "transactions" in chunk:
            combined_json["transactions"].extend(chunk["transactions"])
        else:
            logging.warning(f"Skipping invalid chunk: {chunk}")
    return combined_json

async def process_chunks_in_batches(docs, chunk_size, context, batch_size=10):
    chunks = [docs[i:i + chunk_size] for i in range(0, len(docs), chunk_size)]
    combined_json = {"transactions": []}
    
    # Process chunks in batches
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        try:
            batch_json = await generate_json_chunks(batch, context, financial_prompt)
            combined_json["transactions"].extend(batch_json["transactions"])
        except Exception as e:
            st.error(f"Error processing batch {i // batch_size + 1}: {e}")
            break
    
    return combined_json

def validate_json(json_data):
    try:
        json.loads(json_data)
        return True
    except json.JSONDecodeError:
        return False

def generate_monthly_balances(df_transactions):
    df_transactions['date'] = pd.to_datetime(df_transactions['date'])
    df_transactions = df_transactions.sort_values(by='date')
    df_transactions['month'] = df_transactions['date'].dt.to_period('M')
    last_transactions = df_transactions.groupby('month').last().reset_index()
    last_transactions['month'] = last_transactions['month'].dt.to_timestamp()  # Convert Period to Timestamp
    return last_transactions

async def analyze_transactions_with_llm(transactions):
    context = json.dumps(transactions, indent=2)
    prompt = analysis_prompt.format(context=context)
    analysis_result = await llm.ainvoke(prompt)
    return analysis_result

# Helper function to run async tasks with a progress bar
def run_with_progress_bar(async_func, *args, description="Processing", total_steps=100, **kwargs):
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text(f"{description}... 0%")

    # Simulate progress (you can adjust logic based on actual task completion)
    async def wrapped_func():
        result = await async_func(*args, **kwargs)
        return result

    loop = asyncio.get_event_loop()
    if loop.is_running():
        task = asyncio.run_coroutine_threadsafe(wrapped_func(), loop)
        # Simulate progress updates (adjust based on your async task's nature)
        for i in range(total_steps):
            task_result = task.result() if task.done() else None
            if task_result:
                progress_bar.progress(100)
                status_text.text(f"{description}... 100%")
                break
            progress_bar.progress(i + 1)
            status_text.text(f"{description}... {i + 1}%")
            asyncio.sleep(0.1)  # Simulate async work; adjust timing as needed
        result = task.result()
    else:
        result = loop.run_until_complete(wrapped_func())
        for i in range(total_steps):
            progress_bar.progress(i + 1)
            status_text.text(f"{description}... {i + 1}%")
            asyncio.sleep(0.01)  # Simulate progress; adjust timing
        progress_bar.progress(100)
        status_text.text(f"{description}... 100%")

    # Clear the progress bar and status text after completion
    progress_bar.empty()
    status_text.empty()
    return result

def generate_pie_chart(df, names_col, values_col, title, hole=0.3):
    """
    Generate a Plotly pie chart.
    :param df: DataFrame containing the data.
    :param names_col: Column name to use as labels.
    :param values_col: Column name to use for values.
    :param title: Title of the pie chart.
    :param hole: Size of the donut hole (default 0.3 for donut chart).
    :return: Plotly pie chart figure.
    """
    fig = px.pie(
        df,
        names=names_col,
        values=values_col,
        title=title,
        hole=hole
    )
    return fig

# Streamlit UI
st.set_page_config(page_title="Bank Statement Analyzer", page_icon="💰", layout="centered")
st.title("📊 Bank Statement Analyzer")

# Add a small subtitle/message below the title
st.markdown(
    """
    <div style="font-size: 1rem; font-style: italic; color: #6b7280; margin-top: -10px; margin-bottom: 20px;">
    Kindly upload the bank statements for the past three months. Thank you.
    </div>
    """,
    unsafe_allow_html=True
)

# File uploader
file = st.sidebar.file_uploader("Upload your document", type=["pdf", "txt", "docx", "jpg", "jpeg", "png", "xlsx", "csv"])

# Check if a new file is uploaded by comparing with stored file name
if file:
    if "last_uploaded_file" not in st.session_state:
        st.session_state.last_uploaded_file = None

    # If the file name changes, clear session state and process the new file
    if st.session_state.last_uploaded_file != file.name:
        st.session_state.clear()  # Clear all session state to reset
        st.session_state.last_uploaded_file = file.name

    file_extension = file.name.split(".")[-1].lower()
    docs = parse_file(file, file_extension)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=5)
    final_documents = text_splitter.split_documents(docs)
    
    if not final_documents:
        st.error("No valid text extracted.")
        st.stop()

    if "document_texts" not in st.session_state:
        st.session_state.document_texts = [doc.page_content for doc in final_documents if doc.page_content.strip()]

    document_texts = st.session_state.document_texts
    context = " ".join(document_texts)
    
    if context:
        if "combined_json" not in st.session_state:
            st.session_state.combined_json = run_with_progress_bar(
                process_chunks_in_batches,
                document_texts,
                80,
                context,
                batch_size=10,
                description="Processing transactions in batches"
            )
        
        combined_json = st.session_state.combined_json
        
        if validate_json(json.dumps(combined_json)):
            financial_data = combined_json
        
            if "transactions" in financial_data and financial_data["transactions"]:
                # Apply tagging and flagging
                transactions = financial_data["transactions"]
                transactions = refine_tagging(transactions)
                transactions = flag_recurring_transactions(transactions)
                transactions, weekend_spend, weekday_spend = flag_weekend_vs_weekday(transactions)
                transactions = flag_high_risk_transactions(transactions)
                financial_data["transactions"] = transactions

                # Convert transactions to a DataFrame
                df_transactions = pd.DataFrame(financial_data["transactions"])

                # Ensure 'date' column is in datetime format
                df_transactions['date'] = pd.to_datetime(df_transactions['date'])
                df_transactions['month'] = df_transactions['date'].dt.to_period('M')

                # Calculate AML Risk Score dynamically with enhanced factors
                aml_risk_score = calculate_aml_risk_score(df_transactions)

                # Calculate other metrics for the financial summary card
                daily_avg_balance = df_transactions['balance_remaining'].mean() if not df_transactions.empty else 0
                max_balance = df_transactions['balance_remaining'].max() if not df_transactions.empty else 0
                min_balance = df_transactions['balance_remaining'].min() if not df_transactions.empty else 0

                # Calculate days gap between max and min balance
                if not df_transactions.empty:
                    max_balance_date = df_transactions[df_transactions['balance_remaining'] == max_balance]['date'].iloc[0]
                    min_balance_date = df_transactions[df_transactions['balance_remaining'] == min_balance]['date'].iloc[0]
                    days_gap = abs((max_balance_date - min_balance_date).days)
                else:
                    days_gap = 0

                # Calculate max dormant days (longest period without transactions)
                if len(df_transactions) > 1:
                    df_transactions = df_transactions.sort_values(by='date')
                    date_diffs = df_transactions['date'].diff().dt.days
                    max_dormant_days = int(date_diffs.max()) if not date_diffs.empty else 0
                else:
                    max_dormant_days = 0

                num_transactions = len(df_transactions)

                # Calculate date range
                if not df_transactions.empty:
                    date_range = f"{df_transactions['date'].min().strftime('%d/%m/%Y')}-{df_transactions['date'].max().strftime('%d/%m/%Y')}"
                else:
                    date_range = "N/A"

                # Display the financial summary card
                display_financial_summary_card(
                    aml_risk_score=aml_risk_score,
                    daily_avg_balance=daily_avg_balance,
                    max_balance=max_balance,
                    min_balance=min_balance,
                    days_gap=days_gap,
                    transactions=num_transactions,
                    max_dormant_days=max_dormant_days,
                    date_range=date_range
                )

                # Generate the line chart for monthly balance trend
                last_transactions = generate_monthly_balances(df_transactions)
                fig_monthly_balance_trend = px.line(
                    last_transactions,
                    x='month',
                    y='balance_remaining',
                    title='Monthly Balance Trend'
                )
                fig_monthly_balance_trend.update_traces(texttemplate='%{y:.2f}', textposition='top center')
                fig_monthly_balance_trend.update_layout(
                    xaxis=dict(
                        tickmode='array',
                        tickvals=last_transactions['month'],
                        ticktext=[date.strftime('%Y-%m') for date in last_transactions['month']]
                    )
                )
                st.plotly_chart(fig_monthly_balance_trend)

                # Generate the bar chart for monthly inflow and outflow
                monthly_inflow_outflow = df_transactions.groupby(['month', 'transaction_type'])['amount'].sum().reset_index()
                monthly_inflow_outflow['month'] = monthly_inflow_outflow['month'].astype(str)  # Convert Period to string
                fig_inflow_outflow_bar = px.bar(
                    monthly_inflow_outflow,
                    x='month',
                    y='amount',
                    color='transaction_type',
                    barmode='group',
                    title='Monthly Inflow and Outflow'
                )
                fig_inflow_outflow_bar.update_traces(texttemplate='%{y:.2f}', textposition='outside')
                fig_inflow_outflow_bar.update_layout(
                    xaxis=dict(
                        tickmode='array',
                        tickvals=monthly_inflow_outflow['month'],
                        ticktext=monthly_inflow_outflow['month']
                    )
                )
                st.plotly_chart(fig_inflow_outflow_bar)

                df_debits = df_transactions[df_transactions["transaction_type"] == "Debit"]
                spending_by_category_all = df_debits.groupby("tag")["amount"].sum().reset_index()
                spending_by_category_weekdays = df_debits[~df_debits["is_weekend"]].groupby("tag")["amount"].sum().reset_index()
                spending_by_category_weekends = df_debits[df_debits["is_weekend"]].groupby("tag")["amount"].sum().reset_index()

                # Weekday vs Weekend Spending Pie Chart
                weekday_weekend_spending = df_debits.groupby("is_weekend")["amount"].sum().reset_index()
                weekday_weekend_spending['is_weekend'] = weekday_weekend_spending['is_weekend'].replace({True: 'Weekend', False: 'Weekday'})
                fig_weekday_weekend_pie = generate_pie_chart(
                    weekday_weekend_spending,
                    names_col="is_weekend",
                    values_col="amount",
                    title="Weekday vs Weekend Spending (Total)"
                )
                st.plotly_chart(fig_weekday_weekend_pie)

                # Display totals
                st.write(f"Total Weekend Spending: {weekend_spend:.2f}")
                st.write(f"Total Weekday Spending: {weekday_spend:.2f}")

                # SINGLE radio button for filter
                filter_option = st.radio(
                    "Select view for spending by tag charts (both pie & bar):",
                    ["All Days", "Weekdays Only", "Weekends Only"],
                    horizontal=True
                )

                # Filter data once based on this option
                if filter_option == "All Days":
                    df_debits_filtered = df_debits
                    spending_by_category = spending_by_category_all
                    title_suffix = "(All Days)"
                elif filter_option == "Weekdays Only":
                    df_debits_filtered = df_debits[~df_debits["is_weekend"]]
                    spending_by_category = spending_by_category_weekdays
                    title_suffix = "(Weekdays Only)"
                else:
                    df_debits_filtered = df_debits[df_debits["is_weekend"]]
                    spending_by_category = spending_by_category_weekends
                    title_suffix = "(Weekends Only)"

                # Spending by Tag Pie Chart (controlled by single filter)
                fig_spending_pie = generate_pie_chart(
                    spending_by_category,
                    names_col="tag",
                    values_col="amount",
                    title=f"Spending {title_suffix}"
                )
                st.plotly_chart(fig_spending_pie)

                # Monthly Spending by Tag Bar Chart (also controlled by same filter)
                spending_by_tag_monthly = df_debits_filtered.groupby(
                    ["tag", df_debits_filtered["date"].dt.to_period('M')]
                )["amount"].sum().reset_index()
                spending_by_tag_monthly['date'] = spending_by_tag_monthly['date'].astype(str)

                fig_spending_bar = px.bar(
                    spending_by_tag_monthly,
                    x="date",
                    y="amount",
                    color="tag",
                    title=f"Monthly Spending {title_suffix}"
                )
                st.plotly_chart(fig_spending_bar)

                # Analyze transactions with LLM and store in session state
                if "analysis_result" not in st.session_state:
                    st.session_state.analysis_result = run_with_progress_bar(
                        analyze_transactions_with_llm,
                        financial_data,
                        description="Analyzing transactions"
                    )
                
                st.subheader("Analysis Result")
                st.write(st.session_state.analysis_result.content)

                # Display tagged and flagged transactions
                st.subheader("All Transactions")
                st.dataframe(df_transactions[['date', 'amount', 'transaction_type', 'tag', 'flag', 'is_weekend']])

            else:
                st.warning("No spending data available.")
        else:
            st.error("Invalid JSON")
    else:
        st.error("No text content found.")
else:
    st.warning("Please upload a document.")
