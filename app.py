import os
import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from transformers import pipeline
from datetime import datetime, timedelta
import pytz
import time
import numpy as np
import sqlite3
import streamlit.components.v1 as components

# Load environment variables
load_dotenv()


# Database initialization
def init_db():
    conn = sqlite3.connect('finance_tracker.db')
    c = conn.cursor()

    # Create tables
    c.execute('''
        CREATE TABLE IF NOT EXISTS income (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            amount REAL,
            date TEXT
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS expenses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT,
            amount REAL,
            date TEXT
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS expense_limits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT UNIQUE,
            limit_amount REAL,
            limit_percentage REAL
        )
    ''')

    conn.commit()
    conn.close()


init_db()

# Custom CSS for DeepSeek-inspired theme
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --bg-color: #1e1e2f;
        --primary: #4e73df;
        --secondary: #2d2d44;
        --white: #ffffff;
        --light-gray: #f5f5f5;
        --accent: #ff6b6b;
    }

    /* Overall page styling */
    .stApp {
        background: var(--bg-color) !important;
        color: var(--white) !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Headers */
    h1 {
        color: var(--white) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 2.5em !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        margin-bottom: 30px !important;
    }

    h2, h3 {
        color: var(--white) !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        letter-spacing: 0.3px !important;
    }

    /* Buttons */
    .stButton button {
        background-color: var(--primary) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        padding: 10px 20px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1) !important;
    }

    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }

    /* Input fields */
    .stTextInput input, .stNumberInput input, .stSelectbox, .stDateInput {
        background-color: var(--secondary) !important;
        border: 1px solid var(--primary) !important;
        border-radius: 8px !important;
        font-family: 'Inter', sans-serif !important;
        padding: 8px 12px !important;
        color: var(--white) !important;
    }

    /* Cards */
    .expense-card {
        background: var(--secondary) !important;
        border-radius: 10px !important;
        padding: 20px !important;
        margin: 12px 0 !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1) !important;
        transition: transform 0.3s ease !important;
    }

    .expense-card:hover {
        transform: translateY(-2px) !important;
    }

    /* Chat interface */
    .chat-container {
        background: var(--secondary);
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
    }

    .chat-message {
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 10px;
        max-width: 80%;
    }

    .chat-message.user {
        background: var(--primary);
        margin-left: auto;
    }

    .chat-message.ai {
        background: var(--secondary);
        margin-right: auto;
        border: 1px solid var(--primary);
    }

    /* Custom divider */
    .custom-divider {
        height: 2px;
        background: rgba(255, 255, 255, 0.1);
        margin: 30px 0;
        border-radius: 1px;
    }
</style>
""", unsafe_allow_html=True)


# Database operations
def save_income_to_db(amount, date):
    conn = sqlite3.connect('finance_tracker.db')
    c = conn.cursor()
    c.execute('INSERT INTO income (amount, date) VALUES (?, ?)', (amount, date))
    conn.commit()
    conn.close()


def save_expense_to_db(category, amount, date):
    conn = sqlite3.connect('finance_tracker.db')
    c = conn.cursor()
    c.execute('INSERT INTO expenses (category, amount, date) VALUES (?, ?, ?)',
              (category, amount, date))
    conn.commit()
    conn.close()


def get_current_month_data():
    conn = sqlite3.connect('finance_tracker.db')
    current_month = datetime.now().strftime('%Y-%m')

    # Get income
    c = conn.cursor()
    c.execute('SELECT amount FROM income WHERE date LIKE ?', (f'{current_month}%',))
    income_result = c.fetchone()
    current_income = income_result[0] if income_result else 0

    # Get expenses
    expenses_df = pd.read_sql_query(
        'SELECT category, amount FROM expenses WHERE date LIKE ?',
        conn,
        params=(f'{current_month}%',)
    )

    conn.close()
    return current_income, expenses_df


def save_expense_limit(category, limit_amount, limit_percentage):
    conn = sqlite3.connect('finance_tracker.db')
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO expense_limits 
                 (category, limit_amount, limit_percentage) 
                 VALUES (?, ?, ?)''',
              (category, limit_amount, limit_percentage))
    conn.commit()
    conn.close()


def get_expense_limits():
    conn = sqlite3.connect('finance_tracker.db')
    limits_df = pd.read_sql_query('SELECT * FROM expense_limits', conn)
    conn.close()
    return limits_df


# Initialize Hugging Face pipeline
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="gpt2")


generator = load_model()


def get_advice(context):
    try:
        response = generator(context, max_length=100, num_return_sequences=1)
        return response[0]['generated_text']
    except Exception as e:
        return f"Error generating advice: {str(e)}"


def analyze_finances():
    current_income, expenses_df = get_current_month_data()
    limits_df = get_expense_limits()

    if expenses_df.empty:
        return "No expenses recorded for the current month."

    total_expenses = expenses_df['amount'].sum()
    savings = current_income - total_expenses
    savings_percentage = (savings / current_income * 100) if current_income > 0 else 0

    analysis = []
    analysis.append(f"Monthly Income: ₹{current_income:,.2f}")
    analysis.append(f"Total Expenses: ₹{total_expenses:,.2f}")
    analysis.append(f"Current Savings: ₹{savings:,.2f} ({savings_percentage:.1f}%)")

    if savings_percentage < 30:
        analysis.append("\n⚠️ Warning: Savings below 30% target")
        needed_savings = current_income * 0.3
        reduction_needed = total_expenses - (current_income - needed_savings)
        analysis.append(f"Reduce expenses by ₹{reduction_needed:,.2f} to reach 30% savings target")
    else:
        analysis.append("\n✅ Meeting savings target of 30%")

    category_expenses = expenses_df.groupby('category')['amount'].sum()
    for category, amount in category_expenses.items():
        limit_row = limits_df[limits_df['category'] == category]
        if not limit_row.empty:
            limit = limit_row.iloc[0]['limit_amount']
            if amount > limit:
                analysis.append(f"\n⚠️ Over budget in {category}")
                analysis.append(f"Spent: ₹{amount:,.2f} vs Limit: ₹{limit:,.2f}")

    return "\n".join(analysis)


# Initialize session state
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

# Main UI
st.title("Personal Finance Tracker")
st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

# Collapsible Sidebar with Quick Actions
with st.sidebar:
    st.markdown("""
        <div style='
            background: var(--secondary);
            padding: 0px;
            border-radius: 10px;'>
            <h2 style='text-align: center; margin-bottom: 20px;'>Quick Actions</h2>
        </div>
    """, unsafe_allow_html=True)

    # Quick Action Buttons
    suggested_questions = [
        "Monthly savings tips",
        "Investment basics",
        "Budgeting guide",
        "Expense reduction",
        "Understanding interest",
        "Retirement planning",
    ]

    for question in suggested_questions:
        if st.button(question):
            st.session_state.user_input = question

# AI Advisor Section
st.header("AI Financial Advisor")
user_input = st.text_input("Ask a financial question:", value=st.session_state.user_input)
if st.button("Get Advice") and user_input:
    with st.spinner("Generating advice..."):
        context = f"""
        Financial Context:
        {analyze_finances()}

        User Question:
        {user_input}

        Financial Advice:
        """
        response = get_advice(context)
        st.markdown(f"""
            <div class='chat-container'>
                <div class='chat-message user'>{user_input}</div>
                <div class='chat-message ai'>{response}</div>
            </div>
        """, unsafe_allow_html=True)

st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

# Monthly Income Section
st.header("Monthly Income")
income = st.number_input("Enter your monthly income (₹):", min_value=0.0, step=100.0)
if st.button("Save Income"):
    save_income_to_db(income, datetime.now().strftime('%Y-%m-%d'))
    st.success("Income saved successfully!")

# Display current financial status
current_income, expenses_df = get_current_month_data()
total_expenses = expenses_df['amount'].sum() if not expenses_df.empty else 0
remaining_balance = current_income - total_expenses

st.markdown(f"""
    <div class='expense-card'>
        <div style='font-size: 1.2em; color: var(--white); margin-bottom: 8px;'>Total Income: ₹{current_income:.2f}</div>
        <div style='font-size: 1.2em; color: var(--primary);'>Total Expenses: ₹{total_expenses:.2f}</div>
        <div style='font-size: 1.2em; color: var(--accent);'>Remaining Balance: ₹{remaining_balance:.2f}</div>
    </div>
""", unsafe_allow_html=True)

st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

# Expense Limits Section
st.header("Set Expense Limits")
limit_cols = st.columns(3)
with limit_cols[0]:
    limit_category = st.selectbox("Category", ["Food", "Rent", "Entertainment", "Bills", "Other"])
with limit_cols[1]:
    limit_amount = st.number_input("Monthly Limit (₹)", min_value=0.0, step=100.0)
with limit_cols[2]:
    limit_percentage = st.number_input("Percentage of Income (%)", min_value=0.0, max_value=100.0, step=1.0)

if st.button("Set Limit"):
    save_expense_limit(limit_category, limit_amount, limit_percentage)
    st.success(f"Limit set for {limit_category}")

# Expense Entry Section
st.header("Add New Expense")
expense_cols = st.columns(3)
with expense_cols[0]:
    category = st.selectbox("Expense Category", ["Food", "Rent", "Entertainment", "Bills", "Other"])
with expense_cols[1]:
    amount = st.number_input("Amount (₹)", min_value=0.0, step=1.0, format="%.2f")
with expense_cols[2]:
    expense_date = st.date_input("Date", value=datetime.today())

if st.button("Save Expense"):
    save_expense_to_db(category, amount, expense_date.strftime("%Y-%m-%d"))
    st.success("Expense saved successfully!")

# Financial Analysis Section
st.header("Financial Analysis")
if st.button("Analyze Finances"):
    analysis = analyze_finances()
    st.markdown(f"""
        <div class='chat-container'>
            <div class='chat-message ai'>{analysis}</div>
        </div>
    """, unsafe_allow_html=True)

# Expense Summary and Visualization
if not expenses_df.empty:
    st.header("Expense Summary")
    summary = expenses_df.groupby('category')['amount'].sum().reset_index()
    st.dataframe(summary)

    # Pie Chart
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='#1e1e2f')
    colors = ['#4e73df', '#2d2d44', '#ff6b6b', '#98c4ab', '#a5c49f']
    explode = [0.02] * len(summary)

    patches, texts, autotexts = ax.pie(
        summary['amount'],
        labels=summary['category'],
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        explode=explode,
        wedgeprops={"edgecolor": "white", "linewidth": 2, "alpha": 0.9}
    )

    plt.setp(autotexts, size=9, weight="bold", color="white")
    plt.setp(texts, size=9, color="white")
    ax.set_title("Expense Distribution", color='white', pad=20, fontsize=14, fontweight="bold")

    st.pyplot(fig)

# Exit Button
if st.button("Close Application"):
    st.markdown("""
        <style>
            .stApp {
                background-color: black !important;
                color: white !important;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='black-bg'>Thank you for using the Finance Tracker! Closing...</div>",
                unsafe_allow_html=True)
    time.sleep(2)
    st.stop()