import streamlit as st
import pandas as pd
import numpy as np


def create_glossary_tab():
    """Create a glossary tab explaining financial terms and abbreviations"""
    st.header("Financial Terms Glossary")

    # Add introduction
    st.markdown("""
    This glossary provides explanations for all technical terms, abbreviations, and financial concepts used 
    throughout the Financial Market Prediction System. Use the category tabs below to browse terms or 
    the search box to find specific terms.
    """)

    # Create categories for better organization
    categories = {
        "Technical Indicators": [
            {"term": "MA", "full": "Moving Average",
             "description": "Average price over a specific time period, used to identify trend direction."},
            {"term": "EMA", "full": "Exponential Moving Average",
             "description": "A type of moving average that gives more weight to recent prices."},
            {"term": "RSI", "full": "Relative Strength Index",
             "description": "Momentum oscillator that measures the speed and change of price movements on a scale from 0 to 100."},
            {"term": "MACD", "full": "Moving Average Convergence Divergence",
             "description": "Trend-following momentum indicator showing relationship between two moving averages of a security's price."},
            {"term": "BBands", "full": "Bollinger Bands",
             "description": "Volatility indicator consisting of a middle band (MA) with upper and lower bands based on standard deviation."},
            {"term": "OBV", "full": "On-Balance Volume",
             "description": "Technical indicator that uses volume flow to predict changes in stock price."},
            {"term": "SMA", "full": "Simple Moving Average",
             "description": "The unweighted mean of the previous n data points."},
            {"term": "VWAP", "full": "Volume-Weighted Average Price",
             "description": "Ratio of the value traded to total volume traded over a specific time period."}
        ],
        "Statistical Metrics": [
            {"term": "RMSE", "full": "Root Mean Square Error",
             "description": "Statistical measure of the differences between predicted and observed values."},
            {"term": "R²", "full": "R-squared",
             "description": "Statistical measure representing the proportion of variance in the dependent variable explained by the independent variables."},
            {"term": "CI", "full": "Confidence Interval",
             "description": "Range of values that is likely to contain the true value with a specified probability."},
            {"term": "MAPE", "full": "Mean Absolute Percentage Error",
             "description": "Measure of prediction accuracy as a percentage."},
            {"term": "MAE", "full": "Mean Absolute Error",
             "description": "Average of the absolute differences between predicted and actual values."},
            {"term": "MSE", "full": "Mean Square Error",
             "description": "Average of the squares of the differences between predicted and actual values."}
        ],
        "Financial Terms": [
            {"term": "YoY", "full": "Year-over-Year",
             "description": "Comparison of a statistic for one period to the same period the previous year."},
            {"term": "QoQ", "full": "Quarter-over-Quarter",
             "description": "Comparison of a statistic for one quarter to the previous quarter."},
            {"term": "ETF", "full": "Exchange-Traded Fund",
             "description": "A type of security that tracks an index, sector, commodity, or other asset, but can be purchased or sold on a stock exchange."},
            {"term": "SLA", "full": "Service Level Agreement",
             "description": "A commitment between a service provider and a client, often specifying metrics by which service is measured."},
            {"term": "MoM", "full": "Month-over-Month",
             "description": "Comparison of a statistic for one month to the previous month."},
            {"term": "P/E", "full": "Price-to-Earnings Ratio",
             "description": "Ratio of a company's share price to its earnings per share."},
            {"term": "EPS", "full": "Earnings Per Share",
             "description": "A company's profit divided by the outstanding shares of its common stock."}
        ],
        "Machine Learning Terms": [
            {"term": "GB", "full": "Gradient Boosting",
             "description": "Machine learning technique for regression and classification problems, which produces a prediction model as an ensemble of weak prediction models."},
            {"term": "RF", "full": "Random Forest",
             "description": "Ensemble learning method that constructs multiple decision trees during training."},
            {"term": "GBR", "full": "Gradient Boosting Regression",
             "description": "An implementation of gradient boosting for regression problems."},
            {"term": "LR", "full": "Linear Regression",
             "description": "Statistical approach for modeling relationship between a dependent variable and independent variables."},
            {"term": "ML", "full": "Machine Learning",
             "description": "Type of artificial intelligence that enables systems to learn from data and improve from experience."}
        ],
        "Real Estate Terms": [
            {"term": "ZHVF", "full": "Zillow Home Value Forecast",
             "description": "Zillow's prediction of home values for specific geographic areas."},
            {"term": "ZHVI", "full": "Zillow Home Value Index",
             "description": "A measure of the typical home value for a given region and housing type."},
            {"term": "DOM", "full": "Days On Market",
             "description": "The number of days a property listing is active before going under contract."},
            {"term": "SFR", "full": "Single-Family Residence",
             "description": "A standalone structure designed to be used as a single dwelling unit."},
            {"term": "HOA", "full": "Homeowners Association",
             "description": "Organization in a subdivision, planned community, or condominium responsible for management of common areas."}
        ],
        "Forex Terms": [
            {"term": "PIP", "full": "Point In Percentage",
             "description": "The smallest price move that a given exchange rate can make."},
            {"term": "FX", "full": "Foreign Exchange",
             "description": "The conversion of one currency into another at a specific rate."},
            {"term": "NFP", "full": "Non-Farm Payroll",
             "description": "Monthly report of paid U.S. workers that strongly impacts currency markets."},
            {"term": "ECB", "full": "European Central Bank",
             "description": "The central bank for the euro and administers monetary policy of the eurozone."},
            {"term": "FOMC", "full": "Federal Open Market Committee",
             "description": "Branch of the Federal Reserve that determines monetary policy."}
        ],
        "Cryptocurrency Terms": [
            {"term": "BTC", "full": "Bitcoin",
             "description": "The first and largest cryptocurrency by market capitalization."},
            {"term": "ETH", "full": "Ethereum",
             "description": "A decentralized, open-source blockchain with smart contract functionality."},
            {"term": "HODL", "full": "Hold On for Dear Life",
             "description": "Slang in the cryptocurrency community for holding a cryptocurrency rather than selling it."},
            {"term": "DeFi", "full": "Decentralized Finance",
             "description": "Financial applications built on blockchain technologies."},
            {"term": "NFT", "full": "Non-Fungible Token",
             "description": "Unique digital asset that represents ownership of real-world items like art, video clips, music, etc."}
        ],
        "Commodity Terms": [
            {"term": "WTI", "full": "West Texas Intermediate",
             "description": "A grade of crude oil used as a benchmark in oil pricing."},
            {"term": "Brent", "full": "Brent Crude",
             "description": "A major trading classification of sweet light crude oil that serves as a benchmark price for purchases of oil worldwide."},
            {"term": "COT", "full": "Commitment of Traders",
             "description": "A weekly report showing the positions of different types of traders in U.S. futures markets."},
            {"term": "Backwardation", "full": "Backwardation",
             "description": "Market condition where the price of a futures contract is lower than the spot price."},
            {"term": "Contango", "full": "Contango",
             "description": "Market condition where the price of a futures contract is higher than the spot price."}
        ]
    }

    # Create category tabs
    category_tabs = st.tabs(list(categories.keys()))

    # For each category tab, display the terms
    for i, (category, tab) in enumerate(zip(categories.keys(), category_tabs)):
        with tab:
            # Create a DataFrame for this category
            df = pd.DataFrame(categories[category])

            # Display the table
            st.table(df[["term", "full", "description"]])

    # Add a search function
    st.subheader("Search Terms")
    search_term = st.text_input("Enter a term to search", "")

    if search_term:
        results = []
        for category, terms in categories.items():
            for term in terms:
                if (search_term.lower() in term["term"].lower() or
                        search_term.lower() in term["full"].lower() or
                        search_term.lower() in term["description"].lower()):
                    term_copy = term.copy()
                    term_copy["category"] = category
                    results.append(term_copy)

        if results:
            st.success(f"Found {len(results)} matching terms")
            results_df = pd.DataFrame(results)
            st.table(results_df[["term", "full", "description", "category"]])
        else:
            st.warning("No matching terms found")

    # Add a downloadable CSV of all terms
    all_terms = []
    for category, terms in categories.items():
        for term in terms:
            term_copy = term.copy()
            term_copy["category"] = category
            all_terms.append(term_copy)

    all_terms_df = pd.DataFrame(all_terms)

    st.download_button(
        label="Download Complete Glossary (CSV)",
        data=all_terms_df.to_csv(index=False).encode('utf-8'),
        file_name="financial_terms_glossary.csv",
        mime="text/csv"
    )