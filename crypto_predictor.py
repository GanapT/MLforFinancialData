import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
import time

# Polygon.io API key
POLYGON_API_KEY = "WNvXC2TTYZah3EFclB3w2nyCaKOKzk3R"


def get_polygon_crypto_data(ticker, timespan="day", limit=365):
    """Get cryptocurrency data from Polygon.io"""
    # Calculate date range (last 1 year)
    to_date = datetime.now()
    from_date = to_date - timedelta(days=365)

    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/{timespan}/{from_date.strftime('%Y-%m-%d')}/{to_date.strftime('%Y-%m-%d')}?apiKey={POLYGON_API_KEY}&limit={limit}"

    try:
        response = requests.get(url)
        if response.status_code != 200:
            st.error(f"API returned status code {response.status_code}: {response.text}")
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
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

        return df

    except Exception as e:
        st.error(f"Error retrieving data from Polygon.io: {e}")
        return None


def prepare_crypto_data(df):
    """Prepare cryptocurrency data for modeling"""
    # Create target variable
    df['target'] = df['Close']

    # Create date features
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['day_of_month'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year

    # Create time series features
    df['time_idx'] = range(len(df))

    # Create technical indicators
    # Moving Averages
    for window in [7, 14, 30]:
        if len(df) > window:
            df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()

    # Volatility
    for window in [7, 14, 30]:
        if len(df) > window:
            df[f'Volatility_{window}'] = df['Close'].rolling(window=window).std()

    # Relative Strength Index (RSI)
    if len(df) > 14:
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    if len(df) > 26:
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Price momentum
    for window in [1, 3, 7, 14]:
        if len(df) > window:
            df[f'Return_{window}d'] = df['Close'].pct_change(window)

    # Log returns
    if len(df) > 1:
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

    # Volume indicators
    if 'Volume' in df.columns:
        for window in [7, 14]:
            if len(df) > window:
                df[f'Volume_MA_{window}'] = df['Volume'].rolling(window=window).mean()

        # Volume change
        df['Volume_Change'] = df['Volume'].pct_change()

        # On-balance volume
        df['OBV'] = np.where(
            df['Close'] > df['Close'].shift(1),
            df['Volume'],
            np.where(
                df['Close'] < df['Close'].shift(1),
                -df['Volume'],
                0
            )
        ).cumsum()

    # Drop NaN values
    df.dropna(inplace=True)

    return df


def backtest_crypto_model(df, future_days=30, test_periods=3):
    """Train cryptocurrency prediction model with backtesting for improved accuracy"""
    # Prepare data
    df_processed = prepare_crypto_data(df.copy())

    # Feature columns
    feature_cols = [col for col in df_processed.columns if
                    col not in ['Date', 'timestamp', 'target', 'Open', 'High', 'Low', 'Close', 'Volume']]

    # Perform backtesting
    backtest_results = []

    # Try different historical periods
    for period in range(test_periods):
        # Define test period - last N days
        test_size = 14  # Two weeks
        test_start = len(df_processed) - (period + 1) * test_size
        test_end = len(df_processed) - period * test_size

        if test_start < 60:  # Need enough data to train
            continue

        # Split data
        train_data = df_processed.iloc[:test_start].copy()
        test_data = df_processed.iloc[test_start:test_end].copy()

        # Try different models and parameters
        for model_type in ['RandomForest', 'GradientBoosting']:
            for n_est in [50, 100, 200]:
                for max_depth in [3, 5, 7]:
                    # Train model
                    X_train = train_data[feature_cols]
                    y_train = train_data['target']
                    X_test = test_data[feature_cols]
                    y_test = test_data['target']

                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    # Create and train model
                    if model_type == 'RandomForest':
                        model = RandomForestRegressor(
                            n_estimators=n_est,
                            max_depth=max_depth,
                            random_state=42
                        )
                    else:
                        model = GradientBoostingRegressor(
                            n_estimators=n_est,
                            max_depth=max_depth,
                            learning_rate=0.1,
                            random_state=42
                        )

                    model.fit(X_train_scaled, y_train)

                    # Evaluate
                    y_pred = model.predict(X_test_scaled)
                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

                    # Store results
                    backtest_results.append({
                        'model_type': model_type,
                        'n_estimators': n_est,
                        'max_depth': max_depth,
                        'r2': r2,
                        'rmse': rmse,
                        'mape': mape
                    })

    # Select best model configuration
    if backtest_results:
        # Sort by R² (higher is better)
        sorted_results = sorted(backtest_results, key=lambda x: x['r2'], reverse=True)
        best_params = {
            'model_type': sorted_results[0]['model_type'],
            'n_estimators': sorted_results[0]['n_estimators'],
            'max_depth': sorted_results[0]['max_depth']
        }
    else:
        # Default parameters if backtesting failed
        best_params = {
            'model_type': 'RandomForest',
            'n_estimators': 100,
            'max_depth': 5
        }

    # Train final model with best parameters
    X = df_processed[feature_cols]
    y = df_processed['target']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create model with best parameters
    if best_params['model_type'] == 'RandomForest':
        model = RandomForestRegressor(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            random_state=42
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            learning_rate=0.1,
            random_state=42
        )

    # Train model
    model.fit(X_scaled, y)

    # Make predictions on training data
    train_predictions = model.predict(X_scaled)

    # Calculate metrics
    r2 = r2_score(y, train_predictions)
    rmse = np.sqrt(mean_squared_error(y, train_predictions))
    mape = np.mean(np.abs((y - train_predictions) / y)) * 100

    # Implement walk-forward validation for future predictions
    last_known_date = df['Date'].iloc[-1]
    future_dates = pd.bdate_range(start=last_known_date + timedelta(days=1), periods=future_days)

    # Use last 60 days as the rolling window
    prediction_window = df.copy().tail(60)

    # For storing predictions
    future_prices = []

    # Walk-forward prediction
    for i in range(future_days):
        # Create a new row for prediction
        new_date = future_dates[i]

        # Process the current window to create features
        temp_df = prepare_crypto_data(prediction_window.copy())

        # Get the last row of features
        if len(temp_df) > 0:
            last_features = temp_df.iloc[-1:][feature_cols]

            # Scale features
            last_features_scaled = scaler.transform(last_features)

            # Predict next price
            next_price = model.predict(last_features_scaled)[0]

            # Add realistic noise based on recent volatility
            recent_volatility = prediction_window['Close'].pct_change().std()
            noise_factor = 1 + np.random.normal(0, recent_volatility) * 0.3
            next_price = next_price * noise_factor

            # Store prediction
            future_prices.append(next_price)

            # Create a new row for the next date
            new_row = pd.DataFrame({
                'Date': [new_date],
                'Open': [prediction_window['Close'].iloc[-1]],
                'Close': [next_price],
                'High': [max(prediction_window['Close'].iloc[-1], next_price) * (
                            1 + abs(np.random.normal(0, recent_volatility) * 0.3))],
                'Low': [min(prediction_window['Close'].iloc[-1], next_price) * (
                            1 - abs(np.random.normal(0, recent_volatility) * 0.3))],
                'Volume': [prediction_window['Volume'].mean()] if 'Volume' in prediction_window.columns else [0]
            })

            # Add to the window for next prediction
            prediction_window = pd.concat([prediction_window.iloc[1:], new_row], ignore_index=True)
        else:
            # Fallback if window processing failed
            last_price = prediction_window['Close'].iloc[-1]
            next_price = last_price * (1 + np.random.normal(0, 0.02))
            future_prices.append(next_price)

            # Create a simple new row
            new_row = pd.DataFrame({
                'Date': [new_date],
                'Open': [last_price],
                'Close': [next_price],
                'High': [next_price * 1.01],
                'Low': [next_price * 0.99],
                'Volume': [prediction_window['Volume'].mean()] if 'Volume' in prediction_window.columns else [0]
            })

            # Add to window
            prediction_window = pd.concat([prediction_window.iloc[1:], new_row], ignore_index=True)

    # Create future predictions dataframe
    future_predictions_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Price': future_prices
    })

    # Add confidence intervals that widen with time
    volatility = df['Close'].pct_change().std()
    future_predictions_df['Upper_Bound'] = future_predictions_df['Predicted_Price'] * (
                1 + 1.96 * volatility * np.sqrt(np.arange(1, len(future_predictions_df) + 1) / 5))
    future_predictions_df['Lower_Bound'] = future_predictions_df['Predicted_Price'] * (
                1 - 1.96 * volatility * np.sqrt(np.arange(1, len(future_predictions_df) + 1) / 5))

    # Calculate feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    return {
        'model': model,
        'scaler': scaler,
        'train_predictions': train_predictions,
        'future_predictions': future_predictions_df,
        'feature_importance': feature_importance,
        'r2': r2,
        'rmse': rmse,
        'mape': mape,
        'best_params': best_params,
        'backtest_results': backtest_results
    }


def train_crypto_model(df, future_days=30):
    """Train a cryptocurrency prediction model with backtesting"""
    return backtest_crypto_model(df, future_days)


def create_crypto_chart(df, ticker):
    """Create an interactive chart for cryptocurrency prices"""
    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        specs=[[{"type": "candlestick"}], [{"type": "scatter"}]]
    )

    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Price"
        ),
        row=1, col=1
    )

    # Add moving averages
    for window, color in zip([20, 50], ['blue', 'orange']):
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

    # Add volume as area chart in the second row
    fig.add_trace(
        go.Bar(
            x=df['Date'],
            y=df['Volume'],
            name="Volume",
            marker_color='rgba(0, 128, 255, 0.3)'
        ),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        title=f"{ticker} Price History",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
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

    # Update y-axis titles
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig


def create_crypto_prediction_chart(df, results, ticker):
    """Create an interactive chart for cryptocurrency price predictions"""
    fig = go.Figure()

    # Add historical data
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Close'],
            name="Historical Price",
            line=dict(color="blue", width=2)
        )
    )

    # Add training predictions (show only last portion for clarity)
    display_limit = min(60, len(df))
    fig.add_trace(
        go.Scatter(
            x=df['Date'].iloc[-display_limit:],
            y=results['train_predictions'][-display_limit:],
            name="Model Fit",
            line=dict(color="green", width=1, dash="dash")
        )
    )

    # Add future predictions
    future_df = results['future_predictions']
    fig.add_trace(
        go.Scatter(
            x=future_df['Date'],
            y=future_df['Predicted_Price'],
            name="Predictions",
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
        title=f"{ticker} Price Prediction",
        xaxis_title="Date",
        yaxis_title="Price ($)",
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


def run_crypto_prediction():
    """Main function for cryptocurrency prediction module"""
    st.header("Cryptocurrency Price Prediction")

    # Sidebar controls
    st.sidebar.subheader("Crypto Parameters")
    st.sidebar.markdown("#### Step 1: Select Cryptocurrency")

    # Get popular cryptocurrencies
    popular_cryptos = get_popular_cryptos()

    # Crypto selection method - dropdown or custom input
    selection_method = st.sidebar.radio(
        "Crypto Selection Method",
        ["Popular Cryptocurrencies", "Custom Ticker"],
        key="crypto_selection_method"
    )

    if selection_method == "Popular Cryptocurrencies":
        # Format options for dropdown to include name and symbol
        crypto_options = [f"{name} ({symbol})" for symbol, name in popular_cryptos.items()]
        selected_option = st.sidebar.selectbox("Select Cryptocurrency", crypto_options, key="crypto_select")

        # Extract ticker symbol from selection using rsplit to correctly get the ticker
        ticker = selected_option.rsplit('(', 1)[-1].replace(')', '').strip()
    else:
        # Custom ticker input
        ticker = st.sidebar.text_input("Crypto Ticker Symbol", "X:BTCUSD", key="crypto_ticker")
        st.sidebar.markdown("*Use format 'X:BTCUSD' for Polygon.io API*")

    # Advanced options
    with st.sidebar.expander("Advanced Options", expanded=False):
        timespan = st.selectbox(
            "Data Frequency",
            ["day", "hour", "minute"],
            index=0,
            key="crypto_timespan"
        )
        future_days = st.slider(
            "Days to Predict",
            5, 60, 30,
            key="crypto_future_days"
        )
        test_periods = st.slider(
            "Backtest Periods",
            1, 5, 3,
            key="crypto_test_periods"
        )

    # Initialize session state
    if 'crypto_data' not in st.session_state:
        st.session_state.crypto_data = None
    if 'crypto_results' not in st.session_state:
        st.session_state.crypto_results = None

    # Market overview information
    with st.expander("Cryptocurrency Market Insights", expanded=False):
        st.markdown("""
            ### Key Cryptocurrency Factors

            When analyzing cryptocurrency markets, consider these important factors:

            1. **Market Volatility**: Cryptocurrencies typically exhibit higher volatility than traditional assets. 
               This model accounts for volatility by using wider confidence intervals and incorporating historical 
               price patterns.

            2. **Technical Indicators**: Technical analysis plays a crucial role in crypto trading. Key indicators 
               like RSI, MACD, and moving averages often signal potential trend reversals.

            3. **Market Sentiment**: Cryptocurrency prices can be highly reactive to news events, social media trends, 
               and market sentiment. Our model captures recent momentum but cannot predict unexpected news events.

            4. **Liquidity Considerations**: Different cryptocurrencies have varying levels of liquidity, which affects 
               price stability. Major coins like Bitcoin and Ethereum typically have higher liquidity than smaller altcoins.
            """)

    # Load data button
    col1, col2 = st.columns([3, 1])
    with col2:
        load_data = st.button("Load Crypto Data", key="crypto_load_btn", use_container_width=True)

    if load_data:
        with st.spinner(f"Loading data for {ticker}..."):
            df = get_polygon_crypto_data(ticker, timespan)
            if df is not None and not df.empty:
                st.session_state.crypto_data = df
                st.success(f"Successfully loaded data for {ticker}")

                # Display price chart as soon as data is loaded
                price_chart = create_crypto_chart(df, ticker)
                st.plotly_chart(price_chart, use_container_width=True)

                # Show basic statistics
                with st.expander("Market Statistics", expanded=True):
                    # Calculate key metrics
                    current_price = df['Close'].iloc[-1]
                    prev_price = df['Close'].iloc[-2]
                    daily_change = (current_price - prev_price) / prev_price * 100

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Current Price", f"${current_price:.2f}", f"{daily_change:.2f}%")

                    # All-time high/low
                    high_all = df['High'].max()
                    low_all = df['Low'].min()
                    col2.metric("All-Time High", f"${high_all:.2f}")
                    col3.metric("All-Time Low", f"${low_all:.2f}")

                    # Average volume
                    avg_vol = df['Volume'].mean() if 'Volume' in df.columns else 0
                    col4.metric("Avg. Volume", f"{avg_vol:,.0f}")

                    # More detailed stats
                    st.markdown("---")
                    col1, col2 = st.columns(2)

                    with col1:
                        # Calculate returns
                        monthly_return = (current_price / df['Close'].iloc[-30] - 1) * 100 if len(df) >= 30 else None
                        quarterly_return = (current_price / df['Close'].iloc[-90] - 1) * 100 if len(df) >= 90 else None
                        yearly_return = (current_price / df['Close'].iloc[-365] - 1) * 100 if len(df) >= 365 else None

                        st.markdown("#### Returns")
                        returns_data = pd.DataFrame({
                            'Period': ['Daily', '30-Day', '90-Day', '365-Day'],
                            'Return (%)': [
                                f"{daily_change:.2f}%",
                                f"{monthly_return:.2f}%" if monthly_return is not None else "N/A",
                                f"{quarterly_return:.2f}%" if quarterly_return is not None else "N/A",
                                f"{yearly_return:.2f}%" if yearly_return is not None else "N/A"
                            ]
                        })
                        st.dataframe(returns_data, hide_index=True, use_container_width=True)

                    with col2:
                        # Calculate volatility
                        daily_vol = df['Close'].pct_change().std() * 100
                        monthly_vol = df['Close'].pct_change(30).std() * 100 if len(df) >= 60 else None

                        st.markdown("#### Volatility & Risk")
                        vol_data = pd.DataFrame({
                            'Metric': ['Daily Volatility', 'Monthly Volatility', 'Sharpe Ratio'],
                            'Value': [
                                f"{daily_vol:.2f}%",
                                f"{monthly_vol:.2f}%" if monthly_vol is not None else "N/A",
                                f"{(yearly_return / (daily_vol * np.sqrt(365))):.2f}" if yearly_return is not None else "N/A"
                            ]
                        })
                        st.dataframe(vol_data, hide_index=True, use_container_width=True)
            else:
                st.error(f"Could not retrieve data for {ticker}. Please check the ticker symbol and try again.")

    # Display data and run predictions if data is loaded
    if st.session_state.crypto_data is not None:
        df = st.session_state.crypto_data

        # Train model button
        train_model = st.button("Generate Price Predictions", key="crypto_predict_btn", use_container_width=True)

        if train_model:
            with st.spinner("Training model and generating predictions..."):
                results = backtest_crypto_model(df.copy(), future_days=future_days, test_periods=test_periods)
                st.session_state.crypto_results = results
                st.success("Predictions generated successfully!")

        # Display predictions if available
        if st.session_state.crypto_results is not None:
            results = st.session_state.crypto_results

            # Display performance chart
            st.markdown("### Price Predictions")
            performance_chart = create_crypto_prediction_chart(df, results, ticker)
            st.plotly_chart(performance_chart, use_container_width=True)

            # Display backtesting results
            if 'backtest_results' in results and results['backtest_results']:
                with st.expander("Backtesting Analysis", expanded=False):
                    st.markdown("#### Model Selection Process")
                    st.markdown(f"""
                        The model evaluated {len(results['backtest_results'])} different configurations on historical data
                        to find the optimal prediction strategy. The best configuration was:

                        - **Model Type**: {results['best_params']['model_type']}
                        - **Parameters**: 
                          - n_estimators: {results['best_params']['n_estimators']}
                          - max_depth: {results['best_params']['max_depth']}
                        """)

                    # Show average performance by model type
                    backtest_df = pd.DataFrame(results['backtest_results'])
                    if 'model_type' in backtest_df.columns:
                        # Show average R² for each model type
                        avg_r2 = backtest_df.groupby('model_type')['r2'].mean().reset_index()
                        avg_r2.columns = ['Model Type', 'Average R²']

                        # Create bar chart
                        fig = go.Figure(go.Bar(
                            x=avg_r2['Model Type'],
                            y=avg_r2['Average R²'],
                            marker_color=['#4e8cff', '#ff6b6b']
                        ))

                        fig.update_layout(
                            title="Average Model Performance in Backtests",
                            xaxis_title="Model Type",
                            yaxis_title="Average R² Score",
                            height=300,
                            template="plotly_white"
                        )

                        st.plotly_chart(fig, use_container_width=True)

            # Display feature importance
            with st.expander("Feature Importance Analysis", expanded=False):
                st.markdown("#### Key Price Drivers")
                st.markdown("""
                    This analysis shows which factors had the greatest influence on the cryptocurrency price predictions.
                    Technical indicators, market patterns, and volatility measures often play important roles in crypto markets.
                    """)

                # Get top 10 features
                top_features = results['feature_importance'].head(10)

                # Create bar chart
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

            # Display future price table
            with st.expander("Detailed Price Predictions", expanded=True):
                future_df = results['future_predictions']

                # Format the dataframe
                display_df = future_df.copy()
                display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                display_df = display_df.rename(columns={
                    'Predicted_Price': 'Predicted Price',
                    'Lower_Bound': 'Lower Bound (95% CI)',
                    'Upper_Bound': 'Upper Bound (95% CI)'
                })

                st.dataframe(
                    display_df.style.format({
                        'Predicted Price': '${:.2f}',
                        'Lower Bound (95% CI)': '${:.2f}',
                        'Upper Bound (95% CI)': '${:.2f}'
                    }),
                    hide_index=True,
                    use_container_width=True
                )

                # Add download button
                csv = future_df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions CSV",
                    data=csv,
                    file_name=f"{ticker.replace(':', '_')}_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
    else:
        # Display placeholder if no data is loaded
        st.info("Please load cryptocurrency data to begin analysis")

def get_popular_cryptos():
    """Return a list of popular cryptocurrencies with correct Polygon.io API symbols"""
    return {
        "X:BTCUSD": "Bitcoin (BTC)",  # Added Bitcoin, which was missing
        "X:XRPUSD": "Ripple (XRP)",
        "X:ADAUSD": "Cardano (ADA)",
        "X:SOLUSD": "Solana (SOL)",
        "X:USDCUSD": "USD Coin (USDC)",  # Replaced Dogecoin with a more stable coin
        "X:DOTUSD": "Polkadot (DOT)",
        "X:AVAXUSD": "Avalanche (AVAX)",
        "X:MATICUSD": "Polygon (MATIC)"
    }
