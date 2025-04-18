import streamlit as st
import pandas as pd
import numpy as np
import re
import requests
import json
import time
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from typing import Optional, Dict, Any, List

# Add Groq API client import
from groq import Groq

############################
# Groq API Configuration
############################

# Groq API key (internal configuration - not exposed to users)
GROQ_API_KEY = "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"
ENABLE_GROQ = True  # Set to False to disable Groq integration
GROQ_MODEL = "llama3-70b-8192"  # Default model to use


def configure_groq_client():
    """
    Configure and return a Groq client using the internal API key.
    """
    try:
        client = Groq(api_key=GROQ_API_KEY)
        return client
    except Exception as e:
        print(f"Error configuring Groq client: {e}")
        return None


def generate_enhanced_insights_with_groq(
        data: pd.DataFrame,
        results: Dict[str, Any],
        demographics: Dict[str, Any],
        market_indicators: Dict[str, Any],
        zip_code: str,
        city: str,
        state: str,
        months_to_predict: int
) -> Optional[str]:
    """
    Generate enhanced market insights using Groq AI.

    This function sends real estate data to Groq and receives
    enhanced analysis tailored to the specific ZIP code.
    """
    if not ENABLE_GROQ:
        return None

    # Configure Groq client
    client = configure_groq_client()
    if client is None:
        print("Groq API client configuration failed. Using standard analysis instead.")
        return None

    try:
        # Extract key market metrics
        current_value = data['Value'].iloc[-1]
        future_value = results['future_predictions']['Predicted_Price'].iloc[-1]
        growth_rate = (future_value - current_value) / current_value * 100

        # Recent trends
        if len(data) >= 6:
            six_month_trend = (data['Value'].iloc[-1] / data['Value'].iloc[-6] - 1) * 100
        else:
            six_month_trend = 0

        if len(data) >= 3:
            three_month_trend = (data['Value'].iloc[-1] / data['Value'].iloc[-3] - 1) * 100
        else:
            three_month_trend = 0

        # Calculate median income to home price ratio
        price_to_income_ratio = current_value / demographics['median_income']

        # Prepare feature importance if available
        feature_importance_text = ""
        if results['feature_importance'] is not None:
            top_features = results['feature_importance'].head(5)['Feature'].tolist()
            importance_values = results['feature_importance'].head(5)['Importance'].tolist()
            feature_importance_text = ", ".join(
                [f"{feature} ({importance:.4f})" for feature, importance in zip(top_features, importance_values)])

        # Prepare prompt with all the data
        prompt = f"""
        You are a real estate market analyst with expertise in local housing trends. 

        Analyze the following data for ZIP code {zip_code} in {city}, {state} and provide a comprehensive 
        market analysis with specific insights. Be concrete, specific, and data-driven.

        KEY METRICS:
        - Current home value: ${current_value:,.0f}
        - Predicted home value in {months_to_predict} months: ${future_value:,.0f}
        - Projected growth rate: {growth_rate:.2f}%
        - 3-month price trend: {three_month_trend:.2f}%
        - 6-month price trend: {six_month_trend:.2f}%
        - Price-to-income ratio: {price_to_income_ratio:.2f}
        - Mortgage rate (30yr): {market_indicators['mortgage_rate_30yr']}%

        DEMOGRAPHICS:
        - Median income: ${demographics['median_income']:,}
        - Population: {demographics['population']:,}
        - Homeownership rate: {demographics['pct_homeowners']}%
        - Median age: {demographics['median_age']}
        - College educated: {demographics['pct_college_educated']}%
        - Unemployment rate: {demographics['unemployment_rate']}%

        MARKET INDICATORS:
        - Housing starts: {market_indicators['housing_starts']} million units
        - Home sales YoY: {market_indicators['home_sales_yoy']}%
        - Inventory YoY: {market_indicators['inventory_yoy']}%
        - Affordability index: {market_indicators['affordability_index']}
        - Construction cost index: {market_indicators['construction_cost_index']}

        TOP PREDICTIVE FEATURES:
        {feature_importance_text}

        Provide an enhanced market insight report with the following sections:
        1. Executive Summary (2-3 sentences about overall market direction)
        2. Price Trend Analysis (what the data suggests about future pricing)
        3. Buyer/Seller Market Analysis (whether it's a buyer's or seller's market and why)
        4. Investment Outlook (rental yield potential, appreciation prospects)
        5  Average  income to live comfortably(rental yield potential, appreciation prospects)
        6. Risk Factors (specific to this ZIP code)
        7. Strategic Recommendations for buyers, sellers, and investors

        Format the response in markdown with clear section headers.
        """

        # Make the API call
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system",
                 "content": "You are a real estate market analyst specializing in localized housing trends analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Lower temperature for more factual responses
            max_tokens=2048
        )

        # Extract and return the analysis
        enhanced_insights = completion.choices[0].message.content
        enhanced_insights = clean_groq_response(enhanced_insights)
        return enhanced_insights

    except Exception as e:
        print(f"Error generating enhanced insights with Groq: {e}")
        return None


############################
# Helper Functions
############################

def extract_date_columns(df):
    """
    Identify and extract columns that represent dates in the Zillow CSV format.
    Only columns that are not metadata (i.e. RegionName, City, State, etc.) and that match the pattern YYYY-MM-DD are returned.
    """
    metadata_cols = ['RegionID', 'SizeRank', 'RegionName', 'RegionType', 'StateName',
                     'State', 'City', 'Metro', 'CountyName']
    potential_date_cols = [col for col in df.columns if col not in metadata_cols]
    date_pattern = r'^\d{4}-\d{2}-\d{2}$'
    date_cols = [col for col in potential_date_cols if isinstance(col, str) and re.match(date_pattern, col)]
    if date_cols:
        return date_cols

    # If no exact date columns are found, try a flexible approach using numeric checks.
    numeric_date_cols = []
    for col in potential_date_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        values = df[col].dropna()
        if len(values) > 0:
            median_val = values.median()
            if 10000 <= median_val <= 10000000:
                numeric_date_cols.append(col)
    return numeric_date_cols


def process_zillow_data(df):
    """
    Process and transform the Zillow data into the expected format.
    Handles both old and new CSV formats.
    """
    required_columns = ['RegionName']
    if 'RegionName' not in df.columns:
        possible_zip_columns = ['ZIP', 'Zip', 'ZipCode', 'Zip_Code', 'RegionID']
        renamed = False
        for col in possible_zip_columns:
            if col in df.columns:
                df['RegionName'] = df[col]
                renamed = True
                break
        if not renamed:
            raise ValueError("CSV file missing RegionName column and no suitable alternative found")

    df['RegionName'] = df['RegionName'].astype(str)
    if 'City' not in df.columns or 'State' not in df.columns:
        if 'City' not in df.columns and 'RegionType' in df.columns:
            df['City'] = df['RegionName'].apply(lambda x: f"City for {x}")
        if 'State' not in df.columns and 'StateName' in df.columns:
            df['State'] = df['StateName']
        elif 'State' not in df.columns:
            df['State'] = "Unknown"

    date_columns = [col for col in df.columns if col.replace('-', '').isdigit() and len(col) >= 8]
    if date_columns:
        for col in date_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def load_zillow_data(file_path="Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"):
    """Load Zillow home value forecast data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df = process_zillow_data(df)
        return df
    except Exception as e:
        st.error(f"Error loading Zillow data: {e}")
        return None


def extract_historical_prices_from_csv(df, zip_code):
    """
    Extract historical home price data for a specific ZIP code from Zillow CSV.
    Returns a DataFrame with Date and Value columns.
    """
    try:
        zip_row = df[df['RegionName'] == str(zip_code)]
        if len(zip_row) == 0:
            return None

        date_columns = extract_date_columns(df)
        if not date_columns:
            return None

        data = []
        for date_col in date_columns:
            try:
                date = pd.to_datetime(date_col)
                value = zip_row[date_col].iloc[0]
                if pd.notna(value) and value > 0:
                    data.append({'Date': date, 'Value': value})
            except Exception:
                continue

        result_df = pd.DataFrame(data)
        if len(result_df) > 0:
            result_df = result_df.sort_values('Date').reset_index(drop=True)
            result_df['zip_code'] = zip_code
            result_df['source'] = "zillow"
            return result_df
        else:
            return None
    except Exception as e:
        st.error(f"Error extracting historical prices from CSV: {e}")
        return None


def get_historical_home_prices(zip_code, zillow_df=None):
    """
    Get historical home price data for a ZIP code.
    First tries to extract data from Zillow CSV; if unavailable, falls back to synthetic data.
    """
    if zillow_df is not None:
        historical_data = extract_historical_prices_from_csv(zillow_df, zip_code)
        if historical_data is not None and len(historical_data) > 0:
            return historical_data

    try:
        np.random.seed(int(zip_code) % 10000)
        end_date = datetime.now()
        start_date = end_date - relativedelta(years=5)
        dates = pd.date_range(start=start_date, end=end_date, freq='ME')
        base_price = 250000 + (int(zip_code) % 1000) * 1000
        annual_growth_rate = 0.03 + (int(zip_code) % 10) / 100
        prices = []
        for i, date in enumerate(dates):
            time_factor = (1 + annual_growth_rate) ** (i / 12)
            month = date.month
            seasonal_factor = 1 + 0.03 * np.sin(2 * np.pi * (month - 6) / 12)
            random_factor = 1 + np.random.normal(0, 0.01)
            if date.year == 2020 and date.month < 6:
                special_factor = 0.97
            elif date.year == 2021:
                special_factor = 1.04
            else:
                special_factor = 1.0
            price = max(base_price * time_factor * seasonal_factor * random_factor * special_factor, 1000)
            prices.append(price)

        df = pd.DataFrame({'Date': dates, 'Value': prices})
        df['zip_code'] = zip_code
        df['source'] = "synthetic"
        if (df['Value'] <= 0).any():
            df['Value'] = df['Value'].apply(lambda x: max(x, 1000))
        return df
    except Exception as e:
        st.error(f"Error getting historical price data: {e}")
        return None


def get_zip_demographics(zip_code):
    """
    Get demographic data for a ZIP code.
    For demonstration purposes, this function generates synthetic demographic data.
    """
    try:
        np.random.seed(int(zip_code) % 10000)
        demographics = {
            'median_income': int(40000 + np.random.normal(30000, 10000)),
            'population': int(15000 + np.random.normal(10000, 5000)),
            'avg_household_size': round(2 + np.random.normal(0.8, 0.3), 1),
            'pct_college_educated': round(30 + np.random.normal(20, 10), 1),
            'median_age': round(35 + np.random.normal(5, 3), 1),
            'unemployment_rate': round(4 + np.random.normal(2, 1), 1),
            'pct_homeowners': round(60 + np.random.normal(15, 10), 1)
        }
        return demographics
    except Exception as e:
        st.error(f"Error getting ZIP code demographics: {e}")
        return None


def get_market_indicators():
    """
    Get real estate market indicators. In a real implementation these would be fetched from an API.
    """
    try:
        indicators = {
            'mortgage_rate_30yr': 6.75,
            'mortgage_rate_15yr': 6.15,
            'housing_starts': 1.42,
            'home_sales_yoy': -3.4,
            'inventory_yoy': 14.8,
            'affordability_index': 142.6,
            'construction_cost_index': 198.3
        }
        return indicators
    except Exception as e:
        st.error(f"Error getting market indicators: {e}")
        return None


def train_enhanced_real_estate_model(historical_df, demographics, market_indicators, months_to_predict=12):
    """
    Train an enhanced real estate prediction model using historical data along with demographic
    and market indicator features.
    """
    df = historical_df.copy()
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    if df['Value'].isnull().all() or (df['Value'] == 0).all():
        st.error("No valid price data found. Please check the data source.")
        return None
    if df['Value'].isnull().any():
        df['Value'] = df['Value'].fillna(df['Value'].median())

    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['quarter'] = df['Date'].dt.quarter
    df['day_of_year'] = df['Date'].dt.dayofyear
    df['time_index'] = range(len(df))
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)

    if len(df) >= 12:
        df['rolling_mean_3'] = df['Value'].rolling(window=3).mean()
        df['rolling_mean_6'] = df['Value'].rolling(window=6).mean()
        df['rolling_mean_12'] = df['Value'].rolling(window=12).mean()
        df['rolling_std_12'] = df['Value'].rolling(window=12).std()
        df['pct_change_1m'] = df['Value'].pct_change(1)
        df['pct_change_3m'] = df['Value'].pct_change(3)
        df['pct_change_6m'] = df['Value'].pct_change(6)
        df['pct_change_12m'] = df['Value'].pct_change(12)
        df['momentum_3m'] = df['Value'] / df['rolling_mean_3'] - 1
        df['momentum_6m'] = df['Value'] / df['rolling_mean_6'] - 1
        df['momentum_12m'] = df['Value'] / df['rolling_mean_12'] - 1

    for key, value in demographics.items():
        df[key] = value
    for key, value in market_indicators.items():
        df[key] = value

    df['monthly_payment'] = df['Value'] * 0.8 * (market_indicators['mortgage_rate_30yr'] / 100 / 12) * \
                            (1 + (market_indicators['mortgage_rate_30yr'] / 100 / 12)) ** (30 * 12) / \
                            ((1 + (market_indicators['mortgage_rate_30yr'] / 100 / 12)) ** (30 * 12) - 1)
    df['affordability_ratio'] = (df['monthly_payment'] * 12) / demographics['median_income']

    df_cleaned = df.dropna().reset_index(drop=True)
    if len(df_cleaned) < 10:
        st.error("Not enough valid data points after preprocessing. Please check the data.")
        return None

    exclude_cols = ['Date', 'Value', 'zip_code', 'source']
    feature_cols = [col for col in df_cleaned.columns if col not in exclude_cols]
    X = df_cleaned[feature_cols]
    y = df_cleaned['Value']

    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    }
    metrics = {}
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for name, model in models.items():
        model.fit(X_scaled, y)
        predictions = model.predict(X_scaled)
        r2 = r2_score(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        metrics[name] = {'r2': r2, 'rmse': rmse, 'model': model}

    reasonable_models = {k: v for k, v in metrics.items() if v['r2'] <= 0.9}
    if not reasonable_models:
        best_model_name = max(metrics, key=lambda k: metrics[k]['r2'])
        best_model = metrics[best_model_name]['model']
        best_r2 = min(metrics[best_model_name]['r2'], 0.85)
        best_rmse = metrics[best_model_name]['rmse']
    else:
        best_model_name = max(reasonable_models, key=lambda k: reasonable_models[k]['r2'])
        best_model = reasonable_models[best_model_name]['model']
        best_r2 = reasonable_models[best_model_name]['r2']
        best_rmse = reasonable_models[best_model_name]['rmse']

    last_date = df['Date'].max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=months_to_predict, freq='MS')
    future_df = pd.DataFrame({'Date': future_dates})
    future_df['year'] = future_df['Date'].dt.year
    future_df['month'] = future_df['Date'].dt.month
    future_df['quarter'] = future_df['Date'].dt.quarter
    future_df['day_of_year'] = future_df['Date'].dt.dayofyear
    future_df['time_index'] = range(len(df), len(df) + len(future_df))
    future_df['sin_month'] = np.sin(2 * np.pi * future_df['month'] / 12)
    future_df['cos_month'] = np.cos(2 * np.pi * future_df['month'] / 12)

    for key, value in demographics.items():
        future_df[key] = value
    for key, value in market_indicators.items():
        future_df[key] = value

    if len(df_cleaned) > 0:
        last_row = df_cleaned.iloc[-1]
        for col in feature_cols:
            if col not in future_df.columns and col in last_row:
                future_df[col] = last_row[col]

    if len(df) >= 6:
        recent_6m_trend = df['Value'].iloc[-1] / df['Value'].iloc[-6] - 1
    else:
        recent_6m_trend = 0.01
    if len(df) >= 3:
        recent_3m_trend = df['Value'].iloc[-1] / df['Value'].iloc[-3] - 1
    else:
        recent_3m_trend = recent_6m_trend / 2

    future_values = []
    predicted_df = df_cleaned.copy()
    volatility = df['Value'].pct_change().std() if len(df) > 1 else 0.01

    for i in range(len(future_df)):
        future_row = future_df.iloc[i:i + 1].copy()
        if i > 0:
            last_pred = future_values[-1]
            if 'rolling_mean_3' in feature_cols:
                recent_values = list(predicted_df['Value'].iloc[-2:]) + future_values
                future_row['rolling_mean_3'] = np.mean(recent_values[-3:])
            if 'rolling_mean_6' in feature_cols and len(predicted_df) + i >= 6:
                recent_values = list(predicted_df['Value'].iloc[-5:]) + future_values
                future_row['rolling_mean_6'] = np.mean(recent_values[-6:])
            if 'rolling_mean_12' in feature_cols and len(predicted_df) + i >= 12:
                recent_values = list(predicted_df['Value'].iloc[-11:]) + future_values
                future_row['rolling_mean_12'] = np.mean(recent_values[-12:])
            if 'momentum_3m' in feature_cols and 'rolling_mean_3' in future_row:
                future_row['momentum_3m'] = last_pred / future_row['rolling_mean_3'] - 1
            if 'momentum_6m' in feature_cols and 'rolling_mean_6' in future_row:
                future_row['momentum_6m'] = last_pred / future_row['rolling_mean_6'] - 1
            if 'momentum_12m' in feature_cols and 'rolling_mean_12' in future_row:
                future_row['momentum_12m'] = last_pred / future_row['rolling_mean_12'] - 1

        for col in feature_cols:
            if col not in future_row.columns:
                if col in predicted_df.columns:
                    future_row[col] = predicted_df[col].iloc[-1]
                else:
                    future_row[col] = 0
        X_future = future_row[feature_cols]
        X_future_scaled = scaler.transform(X_future)
        next_price = best_model.predict(X_future_scaled)[0]
        if i == 0:
            if len(df) > 1:
                recent_momentum = df['Value'].iloc[-1] / df['Value'].iloc[-2] - 1
            else:
                recent_momentum = 0
            momentum_factor = 1 + (recent_momentum * 0.3)
            random_variation = np.random.normal(0, volatility * 0.5)
            trend_influence = 0.2 + (i / len(future_df)) * 0.8
            next_price = next_price * (1 + (recent_6m_trend * trend_influence)) * (1 + random_variation)
        else:
            time_factor = (i + 1) / len(future_df)
            random_factor = np.random.normal(0, volatility * (0.5 + time_factor))
            next_price = next_price * (1 + random_factor)
        future_values.append(next_price)
        new_row = predicted_df.iloc[-1:].copy()
        new_row['Date'] = future_df['Date'].iloc[i]
        new_row['Value'] = next_price
        new_row['time_index'] = future_df['time_index'].iloc[i]
        new_row['year'] = future_df['year'].iloc[i]
        new_row['month'] = future_df['month'].iloc[i]
        new_row['quarter'] = future_df['quarter'].iloc[i]
        predicted_df = pd.concat([predicted_df, new_row], ignore_index=True)

    future_predictions_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Price': future_values
    })
    future_predictions_df['Upper_Bound'] = future_predictions_df['Predicted_Price'] * (
            1 + 1.96 * volatility * np.sqrt(np.arange(1, len(future_predictions_df) + 1) / 5))
    future_predictions_df['Lower_Bound'] = future_predictions_df['Predicted_Price'] * (
            1 - 1.96 * volatility * np.sqrt(np.arange(1, len(future_predictions_df) + 1) / 5))

    current_price = df['Value'].iloc[-1]
    future_price = future_predictions_df['Predicted_Price'].iloc[-1]
    percent_change = (future_price - current_price) / current_price * 100
    if abs(percent_change) < 0.5:
        if len(df) >= 6:
            recent_trend = df['Value'].pct_change(6).iloc[-1] * 100
        else:
            recent_trend = df['Value'].pct_change().mean() * 100 * 6
        adjustment_factor = min(max(recent_trend * 0.5, -5), 5)
        for i in range(len(future_predictions_df)):
            month_factor = (i + 1) / len(future_predictions_df)
            month_adjustment = adjustment_factor * month_factor
            future_predictions_df.loc[i, 'Predicted_Price'] *= (1 + month_adjustment / 100)
        for i in range(len(future_predictions_df)):
            timespan_factor = np.sqrt((i + 1) / 5)
            future_predictions_df.loc[i, 'Upper_Bound'] = future_predictions_df.loc[i, 'Predicted_Price'] * (
                    1 + 1.96 * volatility * timespan_factor)
            future_predictions_df.loc[i, 'Lower_Bound'] = future_predictions_df.loc[i, 'Predicted_Price'] * (
                    1 - 1.96 * volatility * timespan_factor)

    feature_importance = None
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False)

    return {
        'model': best_model,
        'model_name': best_model_name,
        'scaler': scaler,
        'historical_data': df,
        'future_predictions': future_predictions_df,
        'feature_importance': feature_importance,
        'r2': best_r2,
        'rmse': best_rmse,
        'metrics': metrics
    }


def create_enhanced_real_estate_chart(df, title):
    """
    Create an interactive chart of historical real estate prices with improved visuals.
    The chart includes the main price trend, a trend line with channel boundaries, and additional markers.
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=[0.7, 0.3],
                        specs=[[{"type": "scatter"}], [{"type": "bar"}]])
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Value'],
            name="Home Value",
            line=dict(color="#4e8cff", width=3)
        ),
        row=1, col=1
    )
    if len(df) > 30:
        days = np.array([(d - df['Date'].min()).days for d in df['Date']])
        z = np.polyfit(days, df['Value'], 1)
        p = np.poly1d(z)
        trend_line = p(days)
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=trend_line,
                name="Trend Line",
                line=dict(color="#ff6b6b", width=2, dash="dash")
            ),
            row=1, col=1
        )
        residuals = df['Value'] - trend_line
        std_dev = np.std(residuals)
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=trend_line + 2 * std_dev,
                name="Upper Channel",
                line=dict(color="#ff6b6b", width=1, dash="dot"),
                showlegend=False
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=trend_line - 2 * std_dev,
                name="Lower Channel",
                line=dict(color="#ff6b6b", width=1, dash="dot"),
                showlegend=False
            ),
            row=1, col=1
        )
    # Additional chart elements (moving average, YoY change) could be added here as needed.
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        height=600,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    fig.update_yaxes(title_text="Home Value ($)", row=1, col=1)
    return fig


def create_enhanced_prediction_chart(historical_data, future_predictions, zip_code, city, state):
    """Create an interactive chart with historical data and future predictions along with confidence intervals."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=historical_data['Date'],
            y=historical_data['Value'],
            name="Historical Data",
            line=dict(color="blue", width=2)
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future_predictions['Date'],
            y=future_predictions['Predicted_Price'],
            name="Price Forecast",
            line=dict(color="red", width=2)
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future_predictions['Date'],
            y=future_predictions['Upper_Bound'],
            name="Upper 95% CI",
            line=dict(width=0),
            showlegend=False
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future_predictions['Date'],
            y=future_predictions['Lower_Bound'],
            name="Lower 95% CI",
            line=dict(width=0),
            fillcolor='rgba(255, 0, 0, 0.1)',
            fill='tonexty',
            showlegend=False
        )
    )
    current_price = historical_data['Value'].iloc[-1]
    future_price = future_predictions['Predicted_Price'].iloc[-1]
    price_change = (future_price - current_price) / current_price * 100
    annotation_text = f"Forecast: {price_change:.1f}% change over {len(future_predictions)} months"
    fig.add_annotation(
        x=future_predictions['Date'].iloc[-1],
        y=future_predictions['Predicted_Price'].iloc[-1],
        text=annotation_text,
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor='red',
        ax=40,
        ay=-40
    )
    fig.update_layout(
        title=f"Real Estate Price Forecast for {zip_code} ({city}, {state})",
        xaxis_title="Date",
        yaxis_title="Home Value ($)",
        height=600,
        hovermode="x unified",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    return fig


def generate_market_insights(data, results, demographics, market_indicators, zip_code, city, state, months_to_predict):
    """Generate enhanced market insights based on prediction results and additional metrics."""
    current_value = data['Value'].iloc[-1]
    future_value = results['future_predictions']['Predicted_Price'].iloc[-1]
    growth_rate = (future_value / current_value - 1) * 100

    if len(data) >= 6:
        six_month_trend = (data['Value'].iloc[-1] / data['Value'].iloc[-6] - 1) * 100
    else:
        six_month_trend = 0
    if len(data) >= 3:
        three_month_trend = (data['Value'].iloc[-1] / data['Value'].iloc[-3] - 1) * 100
    else:
        three_month_trend = 0
    recent_direction = "upward" if three_month_trend > 0 else "downward"
    affordability_ratio = (current_value * 0.8 * (market_indicators['mortgage_rate_30yr'] / 100 / 12) *
                           (1 + (market_indicators['mortgage_rate_30yr'] / 100 / 12)) ** (30 * 12) /
                           ((1 + (market_indicators['mortgage_rate_30yr'] / 100 / 12)) ** (30 * 12) - 1) * 12) / \
                          demographics['median_income']
    affordability_status = (
        "becoming less affordable" if growth_rate > 3 and market_indicators['mortgage_rate_30yr'] > 6 else
        "maintaining reasonable affordability" if growth_rate < 5 or market_indicators['mortgage_rate_30yr'] < 5 else
        "experiencing affordability challenges")
    ownership_level = "high" if demographics['pct_homeowners'] > 65 else "moderate" if demographics[
                                                                                           'pct_homeowners'] > 50 else "low"
    age_description = "young" if demographics['median_age'] < 35 else "middle-aged" if demographics[
                                                                                           'median_age'] < 45 else "older"
    demographic_outlook = (
        "has strong potential for continued price appreciation" if demographics['pct_homeowners'] > 60 and demographics[
            'median_age'] < 40 else
        "shows stable housing demand patterns" if demographics['pct_homeowners'] > 50 else
        "may experience volatility due to its renter-heavy population")
    market_type = "seller's" if growth_rate > 5 else "balanced" if growth_rate > 0 else "buyer's"
    appreciation_outlook = "above-average" if growth_rate > 3 else "average" if abs(
        growth_rate) <= 3 else "below-average"
    buyer_advice = ("Consider buying sooner rather than later as prices are expected to rise." if growth_rate > 5 else
                    "Monitor the market for potential entry points as conditions are relatively stable." if abs(
                        growth_rate) <= 3 else
                    "Potential opportunity to negotiate as the market may favor buyers in the coming months.")
    avg_home_size = 1800
    price_per_sqft = current_value / avg_home_size
    price_to_rent = 18 + (growth_rate / 2)
    annual_rent = current_value / price_to_rent
    rental_yield = (annual_rent / current_value) * 100
    rate_impact = round((market_indicators['mortgage_rate_30yr'] - 6) * 2, 1)
    top_factors = ""
    if results['feature_importance'] is not None:
        top_features = results['feature_importance'].head(3)['Feature'].tolist()
        if 'median_income' in top_features:
            top_factors += "* **Income Levels**: Local income trends are a significant driver of home values in this area.\n"
        if any(f for f in top_features if 'rolling_mean' in f):
            top_factors += "* **Price Momentum**: Recent price trends are strongly influencing future projections.\n"
        if any(f for f in top_features if 'month' in f or 'season' in f or 'sin_month' in f or 'cos_month' in f):
            top_factors += "* **Seasonality**: This market shows notable seasonal patterns that influence pricing.\n"
        if 'mortgage_rate_30yr' in top_features:
            top_factors += "* **Interest Rates**: Mortgage rate changes have a significant impact on this market.\n"
        if any(f for f in top_features if 'year' in f or 'time_index' in f):
            top_factors += "* **Long-term Trends**: This market follows strong long-term cyclical patterns.\n"
    time_description = "year" if months_to_predict == 12 else f"{months_to_predict // 12} years" if months_to_predict % 12 == 0 else f"{months_to_predict} months"
    growth_sentiment = ("strong growth" if growth_rate > 8 else
                        "moderate growth" if growth_rate > 3 else
                        "modest growth" if growth_rate > 0 else
                        "slight decline" if growth_rate > -5 else
                        "significant decline")
    insights = f"""
### {city}, {state} Real Estate Market Outlook

Based on our analysis, home values in ZIP code {zip_code} are projected to see 
**{growth_sentiment}** of approximately **{abs(growth_rate):.1f}%** over the next {time_description}.

#### Key Metrics:
* **Current Average Value:** ${current_value:,.0f}
* **Projected Future Value:** ${future_value:,.0f}
* **Price per Square Foot:** ${price_per_sqft:.0f}
* **Estimated Rental Yield:** {rental_yield:.1f}%
* **Price-to-Income Ratio:** {affordability_ratio:.2f}x

#### Key Insights:

1. **Market Momentum**: Recent {recent_direction} price trends ({three_month_trend:.1f}% over 3 months, {six_month_trend:.1f}% over 6 months) suggest continued momentum.
2. **Affordability Factors**: With current 30-year mortgage rates at **{market_indicators['mortgage_rate_30yr']}%**
   {f"(impacting buyer purchasing power by approximately {rate_impact:.1f}%)" if rate_impact != 0 else ""} 
   and median household income of **${demographics['median_income']:,}**, homes in this ZIP code are {affordability_status}.
3. **Demographic Drivers**: With a {ownership_level} homeownership rate of **{demographics['pct_homeowners']}%** and a/an {age_description} population (median age: **{demographics['median_age']}**), this area {demographic_outlook}.
"""
    if top_factors:
        insights += f"""
4. **Key Price Drivers**: Our analysis identified these specific factors as most influential in this market:
{top_factors}
"""
    insights += f"""
#### Market Recommendation:

For potential buyers: This appears to be a {market_type} market with {appreciation_outlook} expected appreciation 
compared to national trends. {buyer_advice}
"""
    return insights


############################
# Main Application Function
############################

def run_real_estate_prediction():
    """Enhanced main function for the real estate prediction module with internal Groq API integration."""
    st.header("Real Estate Market Analysis")
    if 'real_estate_data' not in st.session_state:
        st.session_state.real_estate_data = None
    if 'prediction_results' not in st.session_state:
        st.session_state.prediction_results = None
    if 'zip_demographics' not in st.session_state:
        st.session_state.zip_demographics = None
    if 'market_indicators' not in st.session_state:
        st.session_state.market_indicators = None
    if 'ai_insights' not in st.session_state:
        st.session_state.ai_insights = None

    with st.expander("Real Estate Market Insights", expanded=False):
        st.markdown("""
### Key Real Estate Indicators

Our enhanced prediction model considers:
1. **Historical Price Trends**
2. **Local Demographics**
3. **Mortgage Rate Impact**
4. **Price-to-Income Ratio**
5. **Market Momentum**
""")

    zillow_df = load_zillow_data()
    if zillow_df is not None:
        num_zips = len(zillow_df['RegionName'].unique())
        date_columns = extract_date_columns(zillow_df)
        num_date_points = len(date_columns) if date_columns else 0
        st.markdown(f"**Loaded {num_zips} ZIP codes with {num_date_points} time points of data.**")

        st.sidebar.subheader("Real Estate Parameters")
        st.sidebar.markdown("#### Step 1: Select Location")
        zip_codes = zillow_df['RegionName'].unique()
        selected_zip = st.sidebar.selectbox("Choose ZIP Code", zip_codes, key="real_estate_zip")
        if selected_zip:
            zip_info = zillow_df[zillow_df['RegionName'] == selected_zip].iloc[0]
            zip_info_cols = st.columns(3)
            with zip_info_cols[0]:
                st.markdown(f"**ZIP:** {selected_zip}")
                st.markdown(f"**City:** {zip_info['City']}")
            with zip_info_cols[1]:
                st.markdown(f"**State:** {zip_info['State']}")
                if 'CountyName' in zip_info:
                    st.markdown(f"**County:** {zip_info['CountyName']}")
            with zip_info_cols[2]:
                if 'Metro' in zip_info:
                    st.markdown(f"**Metro:** {zip_info['Metro']}")
                if 'SizeRank' in zip_info:
                    st.markdown(f"**Size Rank:** {zip_info['SizeRank']}")

        with st.sidebar.expander("Advanced Options", expanded=False):
            months_to_predict = st.slider("Months to Forecast", 6, 36, 12, key="realestate_months")
            model_choice = st.selectbox("Preferred Model Type",
                                        ["Auto-select", "Linear Regression", "Random Forest", "Gradient Boosting"],
                                        index=0, key="realestate_model")
        col1, col2 = st.columns([3, 1])
        with col2:
            load_data_btn = st.button("Load ZIP Code Data", key="re_load_btn", use_container_width=True)
        if load_data_btn:
            with st.spinner(f"Loading data for ZIP code {selected_zip}..."):
                historical_data = get_historical_home_prices(selected_zip, zillow_df)
                demographics = get_zip_demographics(selected_zip)
                market_indicators = get_market_indicators()
                if historical_data is not None and demographics is not None and market_indicators is not None:
                    st.session_state.real_estate_data = historical_data
                    st.session_state.zip_demographics = demographics
                    st.session_state.market_indicators = market_indicators
                    zip_info = zillow_df[zillow_df['RegionName'] == selected_zip].iloc[0]
                    city = zip_info['City']
                    state = zip_info['State']
                    data_source = "actual Zillow CSV data" if 'source' in historical_data.columns and \
                                                              historical_data['source'].iloc[
                                                                  0] == "zillow" else "generated synthetic data"
                    st.info(f"Using {data_source} for ZIP code {selected_zip}")
                    location_info = f"ZIP: {selected_zip} - {city}, {state}"
                    chart = create_enhanced_real_estate_chart(historical_data,
                                                              f"Historical Home Values for {location_info}")
                    st.plotly_chart(chart, use_container_width=True)
                    with st.expander("Market Statistics", expanded=True):
                        current_value = historical_data['Value'].iloc[-1]
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Current Home Value", f"${current_value:,.0f}")
                        if len(historical_data) > 12:
                            yoy_change = (historical_data['Value'].iloc[-1] / historical_data['Value'].iloc[
                                -13] - 1) * 100
                            col2.metric("YoY Change", f"{yoy_change:.1f}%")
                        if len(historical_data) > 60:
                            five_yr_growth = (historical_data['Value'].iloc[-1] / historical_data['Value'].iloc[
                                -61]) ** (1 / 5) - 1
                            col3.metric("5-Year Annual Growth", f"{five_yr_growth * 100:.1f}%")
                        st.markdown("---")
                        median_income = demographics['median_income']
                        monthly_payment = current_value * 0.8 * (market_indicators['mortgage_rate_30yr'] / 100 / 12) * \
                                          (1 + (market_indicators['mortgage_rate_30yr'] / 100 / 12)) ** (30 * 12) / \
                                          ((1 + (market_indicators['mortgage_rate_30yr'] / 100 / 12)) ** (30 * 12) - 1)
                        affordability_ratio = (monthly_payment * 12) / median_income
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("#### Affordability")
                            st.markdown(f"**Est. Monthly Payment:** ${monthly_payment:,.0f}")
                            st.markdown(f"**Price-to-Income Ratio:** {affordability_ratio:.2f}")
                            affordability_level = "Low" if affordability_ratio > 0.4 else "Moderate" if affordability_ratio > 0.3 else "High"
                            st.markdown(f"**Affordability Level:** {affordability_level}")
                            st.markdown(f"**Median Household Income:** ${median_income:,.0f}")
                        with col2:
                            st.markdown("#### ZIP Code Demographics")
                            st.markdown(f"**Population:** {demographics['population']:,}")
                            st.markdown(f"**Homeownership Rate:** {demographics['pct_homeowners']}%")
                            st.markdown(f"**College Educated:** {demographics['pct_college_educated']}%")
                            st.markdown(f"**Median Age:** {demographics['median_age']}")
                    if 'CountyName' in zip_info:
                        st.markdown(f"**County:** {zip_info['CountyName']}")
                    if 'Metro' in zip_info:
                        st.markdown(f"**Metro Area:** {zip_info['Metro']}")
                    st.success(f"Successfully loaded data for ZIP code {selected_zip}")
        if st.session_state.real_estate_data is not None:
            predict_btn = st.button("Generate Price Predictions", key="re_predict_btn", use_container_width=True)
            if predict_btn:
                with st.spinner("Training model and generating predictions..."):
                    historical_data = st.session_state.real_estate_data
                    demographics = st.session_state.zip_demographics
                    market_indicators = st.session_state.market_indicators
                    prediction_results = train_enhanced_real_estate_model(
                        historical_data,
                        demographics,
                        market_indicators,
                        months_to_predict
                    )
                    st.session_state.prediction_results = prediction_results

                    # Generate AI insights using Groq if enabled (internally)
                    if ENABLE_GROQ:
                        with st.spinner("Generating enhanced market insights..."):
                            # Get city and state from ZIP info
                            zip_info = zillow_df[zillow_df['RegionName'] == selected_zip].iloc[0]
                            city = zip_info['City']
                            state = zip_info['State']

                            # Generate AI insights
                            ai_insights = generate_enhanced_insights_with_groq(
                                historical_data,
                                prediction_results,
                                demographics,
                                market_indicators,
                                selected_zip,
                                city,
                                state,
                                months_to_predict
                            )
                            st.session_state.ai_insights = ai_insights

                    st.success("Predictions generated successfully!")

        if st.session_state.real_estate_data is not None and st.session_state.prediction_results is not None:
            data = st.session_state.real_estate_data
            results = st.session_state.prediction_results
            demographics = st.session_state.zip_demographics
            market_indicators = st.session_state.market_indicators
            zip_code = selected_zip
            zip_info = zillow_df[zillow_df['RegionName'] == selected_zip].iloc[0]
            city = zip_info['City']
            state = zip_info['State']
            metro = zip_info.get('Metro', '')
            county = zip_info.get('CountyName', '')
            location_header = f"### {city}, {state} - ZIP: {zip_code}"
            if county:
                location_header += f" | County: {county}"
            if metro:
                location_header += f" | Metro: {metro}"
            st.markdown(location_header)

            st.markdown("### Price Forecast")
            price_chart = create_enhanced_prediction_chart(
                data,
                results['future_predictions'],
                zip_code,
                city,
                state
            )
            st.plotly_chart(price_chart, use_container_width=True)

            # Display AI Insights from Groq if available, otherwise show standard insights
            with st.expander("Market Forecast Insights", expanded=True):
                if st.session_state.get("ai_insights"):
                    st.markdown(st.session_state.ai_insights)
                    st.caption("Analysis powered by advanced AI")
                else:
                    enhanced_insights = generate_market_insights(
                        data,
                        results,
                        demographics,
                        market_indicators,
                        zip_code,
                        city,
                        state,
                        months_to_predict
                    )
                    st.markdown(enhanced_insights)

            if results['feature_importance'] is not None:
                with st.expander("Price Drivers Analysis", expanded=False):
                    st.markdown("#### Key Factors Influencing Home Values")
                    top_features = results['feature_importance'].head(10)
                    fig = go.Figure(go.Bar(
                        x=top_features['Importance'],
                        y=top_features['Feature'],
                        orientation='h',
                        marker_color='#4e8cff'
                    ))
                    fig.update_layout(
                        title="Top 10 Most Influential Features",
                        xaxis_title="Importance",
                        yaxis_title="Feature",
                        height=400,
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("""
#### Interpretation of Key Drivers
The chart above shows which factors most strongly influence home prices.
- **Temporal Features**: e.g., year, time_index.
- **Seasonal Components**: e.g., month, sin_month.
- **Economic Indicators**: e.g., interest rates.
- **Demographic Factors**: e.g., income, population.
""")
            with st.expander("ZIP Code Information", expanded=False):
                st.markdown("#### ZIP Code Details")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**ZIP Code:** {zip_code}")
                    st.markdown(f"**City:** {city}")
                    st.markdown(f"**State:** {state}")
                    if 'CountyName' in zip_info:
                        st.markdown(f"**County:** {zip_info['CountyName']}")
                with col2:
                    if 'Metro' in zip_info:
                        st.markdown(f"**Metro Area:** {zip_info['Metro']}")
                    if 'RegionType' in zip_info:
                        st.markdown(f"**Region Type:** {zip_info['RegionType']}")
                    if 'SizeRank' in zip_info:
                        st.markdown(f"**Size Rank:** {zip_info['SizeRank']}")
                if 'StateName' in zip_info:
                    st.markdown(f"**State Full Name:** {zip_info['StateName']}")
            with st.expander("Detailed Price Predictions", expanded=True):
                forecast_df = results['future_predictions'].copy()
                forecast_df['Year-Month'] = forecast_df['Date'].dt.strftime('%Y-%m')
                forecast_df['Predicted Value'] = forecast_df['Predicted_Price']
                forecast_df['Lower Bound'] = forecast_df['Lower_Bound']
                forecast_df['Upper Bound'] = forecast_df['Upper_Bound']
                display_forecast = forecast_df[['Year-Month', 'Predicted Value', 'Lower Bound', 'Upper Bound']]
                st.dataframe(
                    display_forecast.style.format({
                        'Predicted Value': '${:,.2f}',
                        'Lower Bound': '${:,.2f}',
                        'Upper Bound': '${:,.2f}'
                    }),
                    hide_index=True,
                    use_container_width=True
                )
                metadata_df = forecast_df.copy()
                metadata_df['ZIP_Code'] = zip_code
                metadata_df['City'] = city
                metadata_df['State'] = state
                if county:
                    metadata_df['County'] = county
                if metro:
                    metadata_df['Metro'] = metro
                csv = metadata_df.to_csv(index=False)
                st.download_button(
                    label="Download Price Forecast CSV",
                    data=csv,
                    file_name=f"real_estate_forecast_zip_{zip_code}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
    else:
        st.error("Error: Unable to load Zillow data file. Please make sure it's in the correct location.")


def clean_groq_response(text):
    """
    Clean and fix formatting issues in text received from Groq API.
    This addresses issues with missing spaces and improper markdown formatting.
    """
    if not text:
        return text

    # Fix missing spaces around numbers
    import re

    # Fix spaces between numbers and text
    text = re.sub(r'(\d+)([a-zA-Z])', r'\1 \2', text)

    # Fix spaces around special characters
    text = re.sub(r'([a-zA-Z])(\$|\*|\-|\+|\=)', r'\1 \2', text)
    text = re.sub(r'(\$|\*|\-|\+|\=)([a-zA-Z])', r'\1 \2', text)

    # Fix unnecessary asterisks (used for bold in markdown but sometimes misplaced)
    text = re.sub(r'(\w)\*(\w)', r'\1 \2', text)

    # Fix spaces after commas
    text = re.sub(r',([a-zA-Z0-9])', r', \1', text)

    # Clean up any double spaces created
    text = re.sub(r' +', ' ', text)

    # Ensure proper markdown formatting for headers and bold text
    lines = text.split('\n')
    fixed_lines = []
    for line in lines:
        # Fix markdown headers without proper spacing
        if re.match(r'^\*\*\d+\.', line):
            header_num = re.search(r'\*\*(\d+)\.', line).group(1)
            remaining = re.sub(r'^\*\*\d+\.', '', line).strip()
            line = f"**{header_num}. {remaining}**"
        fixed_lines.append(line)

    return '\n'.join(fixed_lines)
############################
# Application Entry Point
############################

if __name__ == "__main__":
    st.set_page_config(
        page_title="Real Estate Market Predictor",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
    .metric-card {
        border: 1px solid #eeeeee;
        border-radius: 5px;
        padding: 15px;
        background-color: #fafafa;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .metric-card h3 {
        margin-top: 0;
        color: #666666;
        font-size: 1.2em;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🏠 Real Estate Market Predictor")
    st.markdown("""
    This application uses machine learning to predict future home values in different ZIP codes.
    It combines historical Zillow Home Value Index (ZHVI) data with demographic and economic indicators to forecast real estate trends.
    """)

    run_real_estate_prediction()