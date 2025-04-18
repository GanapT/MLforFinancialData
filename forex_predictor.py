import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
from sklearn.metrics import r2_score, mean_squared_error

# Polygon.io API key
POLYGON_API_KEY = "WNvXC2TTYZah3EFclB3w2nyCaKOKzk3R"


def get_forex_data(ticker, timespan="day", limit=365):
    """Get forex data from Polygon.io using ticker symbols like 'FXE' for Currency ETFs"""
    # Calculate date range (last 1 year)
    to_date = datetime.now()
    from_date = to_date - timedelta(days=365)

    # Format ticker for Polygon.io
    # If it contains a slash, it's in the format "XXX/YYY" and needs to be converted to Polygon.io format
    if "/" in ticker:
        base, quote = ticker.split("/")
        formatted_ticker = f"C:{base}{quote}"
    else:
        # Use as-is if it's already properly formatted or is a forex ETF
        formatted_ticker = ticker

    url = f"https://api.polygon.io/v2/aggs/ticker/{formatted_ticker}/range/1/{timespan}/{from_date.strftime('%Y-%m-%d')}/{to_date.strftime('%Y-%m-%d')}?apiKey={POLYGON_API_KEY}&limit={limit}"

    try:
        response = requests.get(url)
        if response.status_code != 200:
            st.error(f"API returned status code {response.status_code}: {response.text}")
            st.info(
                "For forex pairs, use format 'C:EURUSD' or try using Forex ETFs like 'FXE' for Euro, 'FXY' for Yen, etc.")
            return None

        data = response.json()
        if not data.get('results') or len(data.get('results')) == 0:
            st.warning(f"No data available for {ticker}")
            return None

        # Convert results to DataFrame
        df = pd.DataFrame(data['results'])

        # Rename columns
        df.rename(columns={
            't': 'timestamp',
            'o': 'Open',
            'h': 'High',
            'l': 'Low',
            'c': 'Close',
            'v': 'Volume'
        }, inplace=True)

        # Convert timestamp from milliseconds to datetime
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Select and reorder columns
        columns = ['Date', 'Open', 'High', 'Low', 'Close']
        if 'Volume' in df.columns:
            columns.append('Volume')

        df = df[columns]

        return df

    except Exception as e:
        st.error(f"Error retrieving forex data: {e}")
        return None


def backtest_forex_model(df, future_days=30, test_periods=5):
    """Train forex model with backtesting for improved accuracy"""
    # Prepare data
    df_processed = df.copy()
    df_processed['target'] = df_processed['Close']

    # Create date features
    df_processed['day_of_week'] = df_processed['Date'].dt.dayofweek
    df_processed['day_of_month'] = df_processed['Date'].dt.day
    df_processed['month'] = df_processed['Date'].dt.month
    df_processed['year'] = df_processed['Date'].dt.year
    df_processed['quarter'] = df_processed['Date'].dt.quarter

    # Create technical indicators
    # Moving averages
    for window in [5, 10, 20, 50]:
        if len(df_processed) > window:
            df_processed[f'MA_{window}'] = df_processed['Close'].rolling(window=window).mean()

    # Exponential moving averages
    for window in [5, 10, 20]:
        if len(df_processed) > window:
            df_processed[f'EMA_{window}'] = df_processed['Close'].ewm(span=window, adjust=False).mean()

    # Relative Strength Index (RSI)
    if len(df_processed) > 14:
        delta = df_processed['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df_processed['RSI'] = 100 - (100 / (1 + rs))

    # Price momentum & volatility
    for period in [1, 5, 10, 20]:
        if len(df_processed) > period:
            df_processed[f'Return_{period}d'] = df_processed['Close'].pct_change(period)

    for window in [10, 20]:
        if len(df_processed) > window:
            df_processed[f'Volatility_{window}d'] = df_processed['Close'].rolling(window=window).std() / df_processed[
                'Close']

    # Drop NaN values
    df_processed.dropna(inplace=True)

    # Define feature columns
    feature_cols = [col for col in df_processed.columns if
                    col not in ['Date', 'timestamp', 'target', 'Open', 'High', 'Low', 'Close', 'Volume']]

    # Set up backtesting
    backtest_results = []

    # Perform backtesting on multiple historical periods
    for period in range(1, test_periods + 1):
        # Define test window
        test_size = 14  # Two weeks
        test_start = len(df_processed) - period * test_size
        test_end = len(df_processed) - (period - 1) * test_size

        if test_start < 50:  # Need enough data for training
            continue

        # Split data
        train_data = df_processed.iloc[:test_start].copy()
        test_data = df_processed.iloc[test_start:test_end].copy()

        # Get features and target
        X_train = train_data[feature_cols]
        y_train = train_data['target']
        X_test = test_data[feature_cols]
        y_test = test_data['target']

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Try different parameters
        for n_est in [50, 100, 200]:
            for lr in [0.05, 0.1, 0.2]:
                for depth in [3, 5, 7]:
                    # Create model
                    model = GradientBoostingRegressor(
                        n_estimators=n_est,
                        learning_rate=lr,
                        max_depth=depth,
                        random_state=42
                    )

                    # Train model
                    model.fit(X_train_scaled, y_train)

                    # Evaluate
                    y_pred = model.predict(X_test_scaled)
                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                    # Store results
                    backtest_results.append({
                        'params': {
                            'n_estimators': n_est,
                            'learning_rate': lr,
                            'max_depth': depth
                        },
                        'r2': r2,
                        'rmse': rmse
                    })

    # Find best parameters
    if backtest_results:
        # Sort by R² (higher is better)
        sorted_results = sorted(backtest_results, key=lambda x: x['r2'], reverse=True)
        best_params = sorted_results[0]['params']
    else:
        # Default parameters
        best_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5
        }

    # Train final model on all data
    X = df_processed[feature_cols]
    y = df_processed['target']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create and train model
    model = GradientBoostingRegressor(
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth'],
        random_state=42
    )
    model.fit(X_scaled, y)

    # Make predictions on training data
    train_predictions = model.predict(X_scaled)

    # Calculate metrics
    r2 = r2_score(y, train_predictions)
    rmse = np.sqrt(mean_squared_error(y, train_predictions))

    # Generate future predictions using walk-forward validation
    last_date = df['Date'].iloc[-1]
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=future_days)

    # Initialize with recent history
    prediction_window = df.copy().tail(50)
    future_values = []

    # Walk-forward prediction
    for i in range(future_days):
        # Get next date
        next_date = future_dates[i]

        # Process current window
        current_window = prediction_window.copy()
        current_window['target'] = current_window['Close']

        # Add features
        current_window['day_of_week'] = current_window['Date'].dt.dayofweek
        current_window['day_of_month'] = current_window['Date'].dt.day
        current_window['month'] = current_window['Date'].dt.month
        current_window['year'] = current_window['Date'].dt.year
        current_window['quarter'] = current_window['Date'].dt.quarter

        # Add technical indicators
        for window in [5, 10, 20, 50]:
            if len(current_window) > window:
                current_window[f'MA_{window}'] = current_window['Close'].rolling(window=window).mean()

        for window in [5, 10, 20]:
            if len(current_window) > window:
                current_window[f'EMA_{window}'] = current_window['Close'].ewm(span=window, adjust=False).mean()

        if len(current_window) > 14:
            delta = current_window['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            current_window['RSI'] = 100 - (100 / (1 + rs))

        for period in [1, 5, 10, 20]:
            if len(current_window) > period:
                current_window[f'Return_{period}d'] = current_window['Close'].pct_change(period)

        for window in [10, 20]:
            if len(current_window) > window:
                current_window[f'Volatility_{window}d'] = current_window['Close'].rolling(window=window).std() / \
                                                          current_window['Close']

        # Create a new DataFrame with exactly the same features and order as used in training
        last_features = pd.DataFrame(columns=feature_cols)
        last_features.loc[0] = np.nan  # Initialize with NaN

        # Fill with available values from the current window
        for col in feature_cols:
            if col in current_window.columns and not current_window[col].iloc[-1:].isna().all():
                last_features[col] = current_window[col].iloc[-1]
            else:
                # Use mean from training data for missing features
                last_features[col] = X[col].mean()

        # Scale and predict
        last_features_scaled = scaler.transform(last_features)
        next_price = model.predict(last_features_scaled)[0]

        # Add realistic volatility
        volatility = df['Close'].pct_change().std()
        noise = np.random.normal(0, volatility * 0.3)
        next_price = next_price * (1 + noise)

        # Store prediction
        future_values.append(next_price)

        # Create new row for next window
        new_row = pd.DataFrame({
            'Date': [next_date],
            'Open': [prediction_window['Close'].iloc[-1]],
            'High': [next_price * (1 + abs(np.random.normal(0, volatility * 0.2)))],
            'Low': [next_price * (1 - abs(np.random.normal(0, volatility * 0.2)))],
            'Close': [next_price]
        })

        if 'Volume' in prediction_window.columns:
            new_row['Volume'] = prediction_window['Volume'].mean()

        # Update window
        prediction_window = pd.concat([prediction_window.iloc[1:], new_row], ignore_index=True)

    # Create future predictions dataframe
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Price': future_values
    })

    # Add confidence intervals
    daily_volatility = df['Close'].pct_change().std()
    future_df['Upper_Bound'] = future_df['Predicted_Price'] * (
            1 + 1.96 * daily_volatility * np.sqrt(np.arange(1, len(future_df) + 1) / 5))
    future_df['Lower_Bound'] = future_df['Predicted_Price'] * (
            1 - 1.96 * daily_volatility * np.sqrt(np.arange(1, len(future_df) + 1) / 5))

    # Get feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    return {
        'model': model,
        'scaler': scaler,
        'train_predictions': train_predictions,
        'future_predictions': future_df,
        'feature_importance': feature_importance,
        'r2': r2,
        'rmse': rmse,
        'best_params': best_params,
        'backtest_results': backtest_results
    }


def get_forex_pairs():
    """Return a list of forex ETFs and currency-related tickers"""
    return {
        "FXE": "Euro ETF (EUR/USD)",
        "FXY": "Japanese Yen ETF (JPY/USD)",
        "FXB": "British Pound ETF (GBP/USD)",
        "FXF": "Swiss Franc ETF (CHF/USD)",
        "FXC": "Canadian Dollar ETF (CAD/USD)",
        "FXA": "Australian Dollar ETF (AUD/USD)",
        "USDU": "US Dollar Index",
        "UUP": "Bullish USD ETF",
        "UDN": "Bearish USD ETF",
        "C:EURUSD": "EUR/USD Direct Pair",
        "C:GBPUSD": "GBP/USD Direct Pair",
        "C:USDJPY": "USD/JPY Direct Pair",
        "C:USDCAD": "USD/CAD Direct Pair",
        "C:AUDUSD": "AUD/USD Direct Pair",
        "C:USDCHF": "USD/CHF Direct Pair"
    }


def create_forex_chart(df, name):
    """Create an interactive chart for forex data"""
    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        specs=[[{"type": "scatter"}], [{"type": "scatter"}]]
    )

    # Add price line
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Close'],
            name="Exchange Rate",
            line=dict(color="#1F77B4", width=2)
        ),
        row=1, col=1
    )

    # Add moving averages
    for window, color in zip([20, 50], ['orange', 'purple']):
        if len(df) >= window:
            ma = df['Close'].rolling(window=window).mean()
            fig.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=ma,
                    name=f"{window}-Day MA",
                    line=dict(color=color, width=1.5)
                ),
                row=1, col=1
            )

    # Add daily returns
    if len(df) > 1:
        daily_returns = df['Close'].pct_change() * 100

        # Determine colors based on return value
        colors = ['green' if ret >= 0 else 'red' for ret in daily_returns]

        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=daily_returns,
                name="Daily Return (%)",
                mode='lines',
                line=dict(color='purple')
            ),
            row=2, col=1
        )

        # Add zero line
        fig.add_shape(
            type="line",
            x0=df['Date'].iloc[0],
            x1=df['Date'].iloc[-1],
            y0=0,
            y1=0,
            line=dict(color="black", width=1, dash="dash"),
            row=2, col=1
        )

    # Update layout
    fig.update_layout(
        title=f"{name} Exchange Rate",
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

    # Update y-axis labels
    fig.update_yaxes(title_text="Rate", row=1, col=1)
    fig.update_yaxes(title_text="Daily Return (%)", row=2, col=1)

    return fig


def create_forex_prediction_chart(df, results, name):
    """Create an interactive chart for forex predictions"""
    fig = go.Figure()

    # Add historical data
    fig.add_trace(
        go.Scatter(
            x=df['Date'].iloc[-90:],  # Show last 90 days for clarity
            y=df['Close'].iloc[-90:],
            name="Historical",
            line=dict(color="blue", width=2)
        )
    )

    # Add future predictions
    future_df = results['future_predictions']
    fig.add_trace(
        go.Scatter(
            x=future_df['Date'],
            y=future_df['Predicted_Price'],
            name="Forecast",
            line=dict(color="red", width=2)
        )
    )

    # Add confidence intervals
    fig.add_trace(
        go.Scatter(
            x=future_df['Date'],
            y=future_df['Upper_Bound'],
            name="Upper 95% CI",
            line=dict(width=0),
            showlegend=False
        )
    )

    fig.add_trace(
        go.Scatter(
            x=future_df['Date'],
            y=future_df['Lower_Bound'],
            name="Lower 95% CI",
            line=dict(width=0),
            fillcolor='rgba(255, 0, 0, 0.1)',
            fill='tonexty',
            showlegend=False
        )
    )

    # Update layout
    fig.update_layout(
        title=f"{name} Exchange Rate Forecast",
        xaxis_title="Date",
        yaxis_title="Rate",
        height=500,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


def run_forex_prediction():
    """Main function for forex prediction module"""
    st.header("Forex Market Prediction")

    # Sidebar controls
    st.sidebar.subheader("Forex Parameters")
    st.sidebar.markdown("#### Step 1: Select Currency")

    # Get forex pairs
    forex_pairs = get_forex_pairs()

    # Display information about forex symbol format
    st.sidebar.markdown("""
    #### ℹ️ Forex Symbol Format

    Please use one of these formats:
    - Currency ETFs like "FXE" (Euro), "FXY" (Yen)
    - Direct pairs in format "C:EURUSD" (not "EUR/USD")
    """)

    # Create formatted options for the selectbox - include the symbol in the option text
    options = [f"{name} ({symbol})" for symbol, name in forex_pairs.items()]
    selected_option = st.sidebar.selectbox("Select Currency", options, key="forex_select")

    # Extract symbol from the selected option
    selected_symbol = selected_option.split('(')[1].split(')')[0].strip()

    # Advanced options
    with st.sidebar.expander("Advanced Options", expanded=False):
        timespan = st.selectbox(
            "Timeframe",
            ["day", "hour"],
            index=0,
            key="forex_timespan"
        )
        future_days = st.slider(
            "Days to Predict",
            5, 60, 30,
            key="forex_days"
        )
        test_periods = st.slider(
            "Backtest Periods",
            1, 5, 3,
            key="forex_test_periods"
        )

    # Initialize session state
    if 'forex_data' not in st.session_state:
        st.session_state.forex_data = None
    if 'forex_results' not in st.session_state:
        st.session_state.forex_results = None
    if 'forex_symbol' not in st.session_state:
        st.session_state.forex_symbol = None  # Add this line
    if 'forex_name' not in st.session_state:
        st.session_state.forex_name = None  # Add this line

    # Market insights
    with st.expander("Forex Market Insights", expanded=False):
        st.markdown("""
        ### Key Forex Market Factors

        Foreign exchange markets are influenced by several important factors:

        1. **Interest Rate Differentials**: Currency values are strongly influenced by the difference in interest 
           rates between countries. Higher interest rates typically attract foreign capital and strengthen a currency.

        2. **Economic Indicators**: Key economic data like GDP growth, employment figures, inflation rates, and 
           trade balances can drive significant currency movements.

        3. **Central Bank Policies**: Monetary policy decisions and central bank communications often cause 
           immediate and substantial forex market reactions.

        4. **Political Stability and Geopolitical Events**: Political uncertainty or geopolitical tensions can 
           lead to risk-off sentiment and volatility in currency markets.

        5. **Market Sentiment**: Trader positioning and risk appetite can drive shorter-term currency movements, 
           sometimes overshadowing fundamentals.

        Our prediction model incorporates technical indicators and historical patterns but cannot account for 
        unexpected news events or policy changes.
        """)

    # Load data button
    col1, col2 = st.columns([3, 1])
    with col2:
        load_data = st.button("Load Forex Data", key="forex_load", use_container_width=True)

    if load_data:
        with st.spinner(f"Loading data for {selected_symbol}..."):
            df = get_forex_data(selected_symbol, timespan)
            if df is not None and not df.empty:
                st.session_state.forex_data = df
                st.session_state.forex_symbol = selected_symbol

                # Get the display name for this symbol
                display_name = ""
                for symbol, name in forex_pairs.items():
                    if symbol == selected_symbol:
                        display_name = name
                        break

                if not display_name:
                    # If we don't have a display name, use the symbol
                    display_name = selected_symbol

                st.session_state.forex_name = display_name
                st.success(f"Successfully loaded data for {display_name}")

                # Display exchange rate chart immediately
                chart = create_forex_chart(df, display_name)
                st.plotly_chart(chart, use_container_width=True)

                # Show basic statistics
                with st.expander("Market Statistics", expanded=True):
                    # Calculate key metrics
                    current_rate = df['Close'].iloc[-1]
                    prev_rate = df['Close'].iloc[-2]
                    daily_change = (current_rate - prev_rate) / prev_rate * 100

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Current Rate", f"{current_rate:.4f}", f"{daily_change:.2f}%")

                    # Calculate highs and lows
                    high_90d = df['High'].iloc[-90:].max() if len(df) >= 90 else df['High'].max()
                    low_90d = df['Low'].iloc[-90:].min() if len(df) >= 90 else df['Low'].min()

                    col2.metric("90-Day High", f"{high_90d:.4f}")
                    col3.metric("90-Day Low", f"{low_90d:.4f}")

                    # Calculate volatility
                    daily_vol = df['Close'].pct_change().std() * 100
                    col4.metric("Daily Volatility", f"{daily_vol:.2f}%")

                    # More detailed stats
                    st.markdown("---")
                    col1, col2 = st.columns(2)

                    with col1:
                        # Calculate returns over different periods
                        monthly_return = (current_rate / df['Close'].iloc[-22] - 1) * 100 if len(df) >= 22 else None
                        quarterly_return = (current_rate / df['Close'].iloc[-66] - 1) * 100 if len(df) >= 66 else None
                        yearly_return = (current_rate / df['Close'].iloc[-252] - 1) * 100 if len(df) >= 252 else None

                        st.markdown("#### Performance")
                        returns_data = pd.DataFrame({
                            'Period': ['Daily', '1-Month', '3-Month', '1-Year'],
                            'Change (%)': [
                                f"{daily_change:.2f}%",
                                f"{monthly_return:.2f}%" if monthly_return is not None else "N/A",
                                f"{quarterly_return:.2f}%" if quarterly_return is not None else "N/A",
                                f"{yearly_return:.2f}%" if yearly_return is not None else "N/A"
                            ]
                        })
                        st.dataframe(returns_data, hide_index=True, use_container_width=True)

                    with col2:
                        # Calculate trend strength
                        ma_20 = df['Close'].rolling(window=20).mean().iloc[-1] if len(df) >= 20 else None
                        ma_50 = df['Close'].rolling(window=50).mean().iloc[-1] if len(df) >= 50 else None

                        trend = "Bullish" if current_rate > ma_20 > ma_50 else "Bearish" if current_rate < ma_20 < ma_50 else "Neutral"

                        st.markdown("#### Technical Indicators")
                        tech_data = pd.DataFrame({
                            'Indicator': ['Current Trend', 'MA Alignment', 'Daily Range'],
                            'Value': [
                                trend,
                                f"{'Bullish' if ma_20 > ma_50 else 'Bearish'}" if ma_20 is not None and ma_50 is not None else "N/A",
                                f"{((df['High'].iloc[-1] - df['Low'].iloc[-1]) / df['Low'].iloc[-1] * 100):.2f}%"
                            ]
                        })
                        st.dataframe(tech_data, hide_index=True, use_container_width=True)
            else:
                st.error(f"Could not retrieve data for {selected_symbol}. Please try another currency.")

    # Display data and generate predictions if data is loaded
    if 'forex_data' in st.session_state and st.session_state.forex_data is not None:
        df = st.session_state.forex_data
        symbol = st.session_state.forex_symbol
        name = st.session_state.forex_name

        # Train model button
        train_model = st.button("Generate Forex Predictions", key="forex_predict_btn", use_container_width=True)

        if train_model:
            with st.spinner("Training model and generating predictions..."):
                from sklearn.metrics import r2_score, mean_squared_error
                results = backtest_forex_model(df.copy(), future_days=future_days, test_periods=test_periods)
                st.session_state.forex_results = results
                st.success("Predictions generated successfully!")

        # Display predictions if available
        if 'forex_results' in st.session_state and st.session_state.forex_results is not None:
            results = st.session_state.forex_results


            # Show prediction chart
            st.markdown("### Forecast Chart")
            pred_chart = create_forex_prediction_chart(df, results, name)
            st.plotly_chart(pred_chart, use_container_width=True)

            # Display prediction summary
            last_price = df['Close'].iloc[-1]
            future_price = results['future_predictions']['Predicted_Price'].iloc[-1]
            price_change = (future_price - last_price) / last_price * 100
            direction = "increase" if price_change > 0 else "decrease"

            st.markdown(f"""
            ### Forecast Summary

            The model predicts that {name} will {direction} by approximately 
            **{abs(price_change):.2f}%** over the next {future_days} days, from the current rate of 
            **{last_price:.4f}** to approximately **{future_price:.4f}**.

            This forecast is based on historical patterns and technical indicators, with a model 
            confidence (R²) of **{results['r2']:.4f}**.
            """)

            # Show short-term vs long-term outlook
            short_term = results['future_predictions']['Predicted_Price'].iloc[4]  # 5 days
            mid_term = results['future_predictions']['Predicted_Price'].iloc[
                min(len(results['future_predictions']) - 1, 14)]  # 15 days

            col1, col2 = st.columns(2)
            with col1:
                st.metric("5-Day Outlook", f"{short_term:.4f}",
                          f"{((short_term - last_price) / last_price * 100):.2f}%")

            with col2:
                st.metric("15-Day Outlook", f"{mid_term:.4f}", f"{((mid_term - last_price) / last_price * 100):.2f}%")

            # Show feature importance
            st.markdown("### Key Prediction Factors")
            st.write("These are the factors that most influenced the model's predictions:")

            # Display top 10 features
            top_features = results['feature_importance'].head(10)

            # Create a bar chart of feature importance
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=top_features['Importance'],
                    y=top_features['Feature'],
                    orientation='h',
                    marker=dict(color='royalblue')
                )
            )

            fig.update_layout(
                title="Top 10 Influential Factors",
                xaxis_title="Importance Score",
                yaxis_title="Feature",
                height=400,
                template="plotly_white",
                yaxis=dict(autorange="reversed")  # Display highest importance at top
            )

            st.plotly_chart(fig, use_container_width=True)

            # Show model details in expander
            with st.expander("Advanced Model Details", expanded=False):
                st.markdown("#### Model Parameters")
                st.json(results['best_params'])

                st.markdown("#### Backtesting Results")
                if results['backtest_results']:
                    backtest_df = pd.DataFrame([
                        {
                            'n_estimators': r['params']['n_estimators'],
                            'learning_rate': r['params']['learning_rate'],
                            'max_depth': r['params']['max_depth'],
                            'R²': r['r2'],
                            'RMSE': r['rmse']
                        }
                        for r in results['backtest_results'][:5]  # Show top 5 for clarity
                    ])
                    st.dataframe(backtest_df, hide_index=True, use_container_width=True)
                else:
                    st.info("No backtesting results available")

            # Export forecast data
            forecast_df = results['future_predictions'][['Date', 'Predicted_Price', 'Upper_Bound', 'Lower_Bound']]
            forecast_csv = forecast_df.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="Download Forecast Data (CSV)",
                data=forecast_csv,
                file_name=f"{symbol}_forecast.csv",
                mime="text/csv",
                help="Download the complete forecast data as a CSV file"
            )

            # Disclaimer
            st.markdown("""
            ---
            **Disclaimer**: Forecasts are based on historical patterns and technical indicators only. 
            Market conditions can change rapidly due to unexpected news, economic events, or policy changes. 
            This tool is for educational and research purposes only and should not be considered financial advice.
            """)


# Custom CSS for styling
st.markdown("""
<style>
.metric-card {
    border: 1px solid #ddd;
    border-radius: 5px;
    padding: 15px;
    text-align: center;
    box-shadow: 0 0 10px rgba(0,0,0,0.05);
    height: 100%;
}
.metric-card h3 {
    font-size: 16px;
    margin-bottom: 10px;
    color: #555;
}
</style>
""", unsafe_allow_html=True)