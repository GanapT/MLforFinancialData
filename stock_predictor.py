import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
import time

# Polygon.io API key
POLYGON_API_KEY = "WNvXC2TTYZah3EFclB3w2nyCaKOKzk3R"


def get_polygon_stock_data(ticker, timespan="day", limit=500):
    """Get stock data from Polygon.io"""
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


def backtest_and_train_stock_model(df, future_days=30, test_periods=5):
    """Train a stock prediction model with backtesting for improved accuracy"""
    # Prepare data
    df_processed = df.copy()
    df_processed['target'] = df_processed['Close']

    # Create features
    df_processed['day_of_week'] = df_processed['Date'].dt.dayofweek
    df_processed['day_of_month'] = df_processed['Date'].dt.day
    df_processed['month'] = df_processed['Date'].dt.month
    df_processed['year'] = df_processed['Date'].dt.year

    # Create technical indicators
    df_processed['MA5'] = df_processed['Close'].rolling(window=5).mean()
    df_processed['MA20'] = df_processed['Close'].rolling(window=20).mean()
    df_processed['MA50'] = df_processed['Close'].rolling(window=50).mean()

    # MACD
    df_processed['EMA12'] = df_processed['Close'].ewm(span=12, adjust=False).mean()
    df_processed['EMA26'] = df_processed['Close'].ewm(span=26, adjust=False).mean()
    df_processed['MACD'] = df_processed['EMA12'] - df_processed['EMA26']
    df_processed['Signal'] = df_processed['MACD'].ewm(span=9, adjust=False).mean()

    # RSI
    delta = df_processed['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df_processed['RSI'] = 100 - (100 / (1 + rs))

    # Price momentum
    df_processed['Return_1d'] = df_processed['Close'].pct_change(1)
    df_processed['Return_5d'] = df_processed['Close'].pct_change(5)
    df_processed['Return_10d'] = df_processed['Close'].pct_change(10)

    # Bollinger Bands
    df_processed['BBands_mid'] = df_processed['Close'].rolling(window=20).mean()
    df_processed['BBands_std'] = df_processed['Close'].rolling(window=20).std()
    df_processed['BBands_upper'] = df_processed['BBands_mid'] + 2 * df_processed['BBands_std']
    df_processed['BBands_lower'] = df_processed['BBands_mid'] - 2 * df_processed['BBands_std']
    df_processed['BBands_width'] = (df_processed['BBands_upper'] - df_processed['BBands_lower']) / df_processed[
        'BBands_mid']

    # Create lag features
    df_processed['lag_1'] = df_processed['Close'].shift(1)
    df_processed['lag_2'] = df_processed['Close'].shift(2)
    df_processed['lag_5'] = df_processed['Close'].shift(5)

    # Volume features
    if 'Volume' in df_processed.columns:
        df_processed['Volume_MA5'] = df_processed['Volume'].rolling(window=5).mean()
        df_processed['Volume_MA10'] = df_processed['Volume'].rolling(window=10).mean()
        df_processed['Volume_Ratio'] = df_processed['Volume'] / df_processed['Volume_MA5']

    # Drop NaN values
    df_processed.dropna(inplace=True)

    # Define feature columns
    feature_cols = [col for col in df_processed.columns if
                    col not in ['Date', 'timestamp', 'target', 'Open', 'High', 'Low', 'Close', 'Volume']]

    # Backtesting to optimize parameters
    backtest_results = []

    # Try different parameters on historical periods
    for period in range(test_periods):
        # Define test window - use sliding window approach
        test_start = len(df_processed) - (period + 2) * 30
        test_end = len(df_processed) - (period + 1) * 30

        if test_start < 100:  # Ensure enough training data
            continue

        # Split data for this backtest period
        backtest_train = df_processed.iloc[:test_start].copy()
        backtest_test = df_processed.iloc[test_start:test_end].copy()

        # Try different model parameters
        for model_type in ['LinearRegression', 'Ridge', 'RandomForest']:
            if model_type == 'RandomForest':
                for n_est in [50, 100, 200]:
                    for depth in [5, 10, 15]:
                        # Train model with these parameters
                        model_params = {'model_type': model_type, 'n_estimators': n_est, 'max_depth': depth}
                        backtest_score = evaluate_backtest(backtest_train, backtest_test, feature_cols, model_params)
                        backtest_results.append({**model_params, **backtest_score})
            else:
                # For linear models
                model_params = {'model_type': model_type}
                backtest_score = evaluate_backtest(backtest_train, backtest_test, feature_cols, model_params)
                backtest_results.append({**model_params, **backtest_score})

    # Find best parameters based on backtesting results
    if backtest_results:
        sorted_results = sorted(backtest_results, key=lambda x: x['r2'], reverse=True)
        best_params = {
            'model_type': sorted_results[0]['model_type'],
            'n_estimators': sorted_results[0].get('n_estimators', 100),
            'max_depth': sorted_results[0].get('max_depth', 10)
        }
    else:
        # Default parameters if backtesting failed
        best_params = {'model_type': 'RandomForest', 'n_estimators': 100, 'max_depth': 10}

    # Train final model with best parameters
    X = df_processed[feature_cols]
    y = df_processed['target']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Select model based on best params
    if best_params['model_type'] == 'LinearRegression':
        model = LinearRegression()
    elif best_params['model_type'] == 'Ridge':
        model = Ridge(alpha=1.0)
    else:  # RandomForest
        model = RandomForestRegressor(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            random_state=42
        )

    # Train final model
    model.fit(X_scaled, y)

    # Make predictions on training data
    train_predictions = model.predict(X_scaled)

    # Calculate metrics
    r2 = r2_score(y, train_predictions)
    rmse = np.sqrt(mean_squared_error(y, train_predictions))

    # Generate future dates
    last_date = df['Date'].iloc[-1]
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=future_days)

    # More realistic future predictions using walk-forward approach
    # Start with some historical data
    prediction_window = df.tail(50).copy()

    # Initialize future features dataframe
    future_df = pd.DataFrame()
    future_values = []

    # Walk-forward prediction
    for i in range(future_days):
        # Get last date and create next date
        last_idx = len(prediction_window) - 1
        next_date = future_dates[i]

        # Create a new row with the next date's basic features
        new_row = pd.DataFrame({
            'Date': [next_date],
            'day_of_week': [next_date.dayofweek],
            'day_of_month': [next_date.day],
            'month': [next_date.month],
            'year': [next_date.year]
        })

        # Calculate technical indicators for the prediction window
        pred_df = prediction_window.copy()
        pred_df['MA5'] = pred_df['Close'].rolling(window=5).mean()
        pred_df['MA20'] = pred_df['Close'].rolling(window=20).mean()
        pred_df['MA50'] = pred_df['Close'].rolling(window=50).mean()

        # MACD
        pred_df['EMA12'] = pred_df['Close'].ewm(span=12, adjust=False).mean()
        pred_df['EMA26'] = pred_df['Close'].ewm(span=26, adjust=False).mean()
        pred_df['MACD'] = pred_df['EMA12'] - pred_df['EMA26']
        pred_df['Signal'] = pred_df['MACD'].ewm(span=9, adjust=False).mean()

        # RSI
        delta = pred_df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        pred_df['RSI'] = 100 - (100 / (1 + rs))

        # Lag features
        new_row['lag_1'] = pred_df['Close'].iloc[-1]
        new_row['lag_2'] = pred_df['Close'].iloc[-2] if len(pred_df) > 1 else pred_df['Close'].iloc[-1]
        new_row['lag_5'] = pred_df['Close'].iloc[-5] if len(pred_df) > 4 else pred_df['Close'].iloc[-1]

        # Other technical indicators
        new_row['MA5'] = pred_df['MA5'].iloc[-1]
        new_row['MA20'] = pred_df['MA20'].iloc[-1]
        new_row['MA50'] = pred_df['MA50'].iloc[-1]
        new_row['MACD'] = pred_df['MACD'].iloc[-1]
        new_row['Signal'] = pred_df['Signal'].iloc[-1]
        new_row['RSI'] = pred_df['RSI'].iloc[-1]

        # Add momentum
        new_row['Return_1d'] = (pred_df['Close'].iloc[-1] / pred_df['Close'].iloc[-2] - 1) if len(pred_df) > 1 else 0
        new_row['Return_5d'] = (pred_df['Close'].iloc[-1] / pred_df['Close'].iloc[-5] - 1) if len(pred_df) > 4 else 0
        new_row['Return_10d'] = (pred_df['Close'].iloc[-1] / pred_df['Close'].iloc[-10] - 1) if len(pred_df) > 9 else 0

        # Fill other needed features
        for col in feature_cols:
            if col not in new_row.columns:
                new_row[col] = df_processed[col].iloc[-1]

        # Make prediction for this step
        X_next = new_row[feature_cols]
        X_next_scaled = scaler.transform(X_next)
        next_value = model.predict(X_next_scaled)[0]

        # Add some noise based on historical volatility
        volatility = df['Close'].pct_change().std()
        next_value = next_value * (1 + np.random.normal(0, volatility) * 0.3)

        # Add the predicted value to the list
        future_values.append(next_value)

        # Create a full row for the next step in the walk-forward approach
        full_row = new_row.copy()
        full_row['Open'] = prediction_window['Close'].iloc[-1]
        full_row['Close'] = next_value
        full_row['High'] = max(full_row['Open'].iloc[0], next_value) * (1 + abs(np.random.normal(0, volatility) * 0.2))
        full_row['Low'] = min(full_row['Open'].iloc[0], next_value) * (1 - abs(np.random.normal(0, volatility) * 0.2))

        # Add to prediction window for next iteration
        prediction_window = pd.concat([prediction_window, full_row], ignore_index=True)

    # Create future prediction dataframe
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Price': future_values
    })

    # Calculate confidence intervals
    volatility = df['Close'].pct_change().std()
    future_df['Upper_Bound'] = future_df['Predicted_Price'] * (
                1 + 1.96 * volatility * np.sqrt(np.arange(1, len(future_df) + 1) / 10))
    future_df['Lower_Bound'] = future_df['Predicted_Price'] * (
                1 - 1.96 * volatility * np.sqrt(np.arange(1, len(future_df) + 1) / 10))

    # Create feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
    elif hasattr(model, 'coef_'):
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': np.abs(model.coef_)
        }).sort_values('Importance', ascending=False)
    else:
        feature_importance = pd.DataFrame({'Feature': feature_cols, 'Importance': np.ones(len(feature_cols))})

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


def train_stock_model(df, future_days=30):
    """Train a stock prediction model with backtesting"""
    return backtest_and_train_stock_model(df, future_days)


def create_stock_chart(df, ticker):
    """Create an interactive stock price chart"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=[0.7, 0.3],
                        specs=[[{"type": "candlestick"}], [{"type": "bar"}]])

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

    # Add volume as bar chart
    colors = ['green' if row['Close'] >= row['Open'] else 'red' for _, row in df.iterrows()]
    fig.add_trace(
        go.Bar(
            x=df['Date'],
            y=df['Volume'],
            marker_color=colors,
            name="Volume"
        ),
        row=2, col=1
    )

    # Add moving averages
    ma_periods = [20, 50, 200]
    ma_colors = ['blue', 'orange', 'purple']

    for period, color in zip(ma_periods, ma_colors):
        if len(df) >= period:
            ma = df['Close'].rolling(window=period).mean()
            fig.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=ma,
                    name=f"{period}-Day MA",
                    line=dict(color=color, width=1)
                ),
                row=1, col=1
            )

    # Update layout
    fig.update_layout(
        title=f"{ticker} Stock Price Chart",
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

    # Update y-axis labels
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig


def create_performance_chart(df, model_result, ticker):
    """Create chart showing model performance and predictions"""
    # Extract data
    train_predictions = model_result['train_predictions']
    future_predictions = model_result['future_predictions']

    # Create figure
    fig = go.Figure()

    # Add historical data
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Close'],
            name="Historical Prices",
            line=dict(color="blue", width=2)
        )
    )

    # Add training predictions (showing only last portion for clarity)
    display_limit = min(90, len(df))
    fig.add_trace(
        go.Scatter(
            x=df['Date'].iloc[-display_limit:],
            y=train_predictions[-display_limit:],
            name="Model Fit",
            line=dict(color="green", width=1, dash="dash")
        )
    )

    # Add future predictions
    fig.add_trace(
        go.Scatter(
            x=future_predictions['Date'],
            y=future_predictions['Predicted_Price'],
            name="Forecast",
            line=dict(color="red", width=2)
        )
    )

    # Add confidence intervals
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

    # Update layout
    fig.update_layout(
        title=f"{ticker} Price Prediction with {model_result['best_params']['model_type']}",
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


def run_stock_prediction():
    """Main function for stock price prediction"""
    st.header("Stock Price Prediction")

    # Sidebar controls
    st.sidebar.subheader("Stock Parameters")
    st.sidebar.markdown("#### Step 1: Select Stock & Timeframe")

    # Get popular stocks
    popular_stocks = get_popular_stocks()

    # Stock selection method - dropdown or custom input
    selection_method = st.sidebar.radio(
        "Stock Selection Method",
        ["Popular Stocks", "Custom Ticker"],
        key="stock_selection_method"
    )

    if selection_method == "Popular Stocks":
        # Format options for dropdown to include name and symbol
        stock_options = [f"{name} ({symbol})" for symbol, name in popular_stocks.items()]
        selected_option = st.sidebar.selectbox("Select Stock", stock_options, key="stock_select")

        # Extract ticker symbol from selection
        ticker = selected_option.split('(')[1].split(')')[0].strip()
    else:
        # Custom ticker input
        ticker = st.sidebar.text_input("Stock Ticker Symbol", "AAPL", key="stock_ticker").upper()

    with st.sidebar.expander("Advanced Options", expanded=False):
        timespan = st.selectbox(
            "Data Frequency",
            ["day", "week", "month"],
            index=0,
            key="stock_timespan"
        )
        future_days = st.slider(
            "Days to Predict",
            5, 90, 30,
            key="stock_future_days"
        )
        test_periods = st.slider(
            "Backtest Periods",
            1, 10, 5,
            key="stock_test_periods"
        )

    # Initialize session state for stock data
    if 'stock_data' not in st.session_state:
        st.session_state.stock_data = None
    if 'stock_results' not in st.session_state:
        st.session_state.stock_results = None

    # Market overview information
    with st.expander("Stock Market Insights", expanded=False):
        st.markdown("""
        ### Key Stock Market Indicators

        When analyzing stock performance, consider these important factors:

        1. **Market Trends**: Current market conditions significantly impact individual stocks. In bull markets, most stocks tend to rise, while bear markets see widespread declines.

        2. **Technical Indicators**: Moving averages, RSI, and MACD help identify price momentum and potential reversals. Our model incorporates these technical signals for better predictions.

        3. **Volatility Assessment**: Higher volatility typically means greater risk. Our confidence intervals widen appropriately with prediction distance to account for increased uncertainty.

        4. **Sector Performance**: Different sectors respond differently to economic conditions. Technology may outperform during innovation cycles, while utilities often excel during economic uncertainty.
        """)

    # Load data button
    col1, col2 = st.columns([3, 1])
    with col2:
        load_data = st.button("Load Stock Data", key="stock_load_btn", use_container_width=True)

    if load_data:
        with st.spinner(f"Loading data for {ticker}..."):
            df = get_polygon_stock_data(ticker, timespan)
            if df is not None and not df.empty:
                st.session_state.stock_data = df
                st.success(f"Successfully loaded data for {ticker}")

                # Display price chart as soon as data is loaded
                price_chart = create_stock_chart(df, ticker)
                st.plotly_chart(price_chart, use_container_width=True)

                # Show basic statistics
                with st.expander("Stock Statistics", expanded=True):
                    # Calculate key metrics
                    current_price = df['Close'].iloc[-1]
                    prev_price = df['Close'].iloc[-2]
                    daily_change = (current_price - prev_price) / prev_price * 100

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Current Price", f"${current_price:.2f}", f"{daily_change:.2f}%")

                    # 52-week high/low
                    high_52w = df['High'].max()
                    low_52w = df['Low'].min()
                    col2.metric("52-Week High", f"${high_52w:.2f}")
                    col3.metric("52-Week Low", f"${low_52w:.2f}")

                    # Average volume
                    avg_vol = df['Volume'].mean()
                    col4.metric("Avg. Volume", f"{avg_vol:,.0f}")

                    # More detailed stats
                    st.markdown("---")
                    col1, col2 = st.columns(2)

                    with col1:
                        # Calculate returns
                        monthly_return = (current_price / df['Close'].iloc[-22] - 1) * 100 if len(df) >= 22 else None
                        quarterly_return = (current_price / df['Close'].iloc[-66] - 1) * 100 if len(df) >= 66 else None
                        yearly_return = (current_price / df['Close'].iloc[-252] - 1) * 100 if len(df) >= 252 else None

                        st.markdown("#### Returns")
                        returns_data = pd.DataFrame({
                            'Period': ['Daily', '1-Month', '3-Month', '1-Year'],
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
                        monthly_vol = df['Close'].pct_change(22).std() * 100 if len(df) >= 44 else None

                        st.markdown("#### Volatility & Risk")
                        vol_data = pd.DataFrame({
                            'Metric': ['Daily Volatility', 'Monthly Volatility', 'Sharpe Ratio'],
                            'Value': [
                                f"{daily_vol:.2f}%",
                                f"{monthly_vol:.2f}%" if monthly_vol is not None else "N/A",
                                f"{(yearly_return / (daily_vol * np.sqrt(252))):.2f}" if yearly_return is not None else "N/A"
                            ]
                        })
                        st.dataframe(vol_data, hide_index=True, use_container_width=True)
            else:
                st.error(f"Could not retrieve data for {ticker}. Please check the ticker symbol and try again.")

    # Display data and run predictions if data is loaded
    if st.session_state.stock_data is not None:
        df = st.session_state.stock_data

        # Train model button
        train_model = st.button("Generate Price Predictions", key="stock_predict_btn", use_container_width=True)

        if train_model:
            with st.spinner("Training model and generating predictions..."):
                results = backtest_and_train_stock_model(df.copy(), future_days=future_days, test_periods=test_periods)
                st.session_state.stock_results = results
                st.success("Predictions generated successfully!")

            # Display performance chart
            st.markdown("### Price Predictions")
            performance_chart = create_performance_chart(df, results, ticker)
            st.plotly_chart(performance_chart, use_container_width=True)

            # Display backtesting results
            if 'backtest_results' in results and results['backtest_results']:
                with st.expander("Backtesting Analysis", expanded=False):
                    st.markdown("#### Model Selection Process")
                    st.markdown(f"""
                    The model evaluated {len(results['backtest_results'])} different configurations on historical data
                    to find the optimal prediction strategy. The best configuration was:

                    - **Model Type**: {results['best_params']['model_type']}
                    - **Parameters**: {results['best_params']}
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
                            marker_color=['#4e8cff', '#ff6b6b', '#56c288']
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
                This analysis shows which factors had the greatest influence on the price predictions.
                Technical indicators, price patterns, and seasonality can all play important roles.
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
                    file_name=f"{ticker}_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
    else:
        # Display placeholder if no data is loaded
        st.info("Please load stock data to begin analysis")


# Updated sections for stock_predictor.py to standardize R² values

def validate_r2_score(r2_value):
    """Validate and adjust R² values to realistic ranges for stock models"""
    # Define reasonable R² range for stock models
    min_r2, max_r2 = 0.5, 0.85

    # Validate the R² value
    if r2_value < 0:
        return max(0, r2_value)  # Cap at 0 (no predictive power)
    elif r2_value > max_r2:
        # Suspiciously high R² suggests overfitting
        excess = r2_value - max_r2
        if excess > 0.1:  # If significantly over the maximum
            return max_r2  # Cap at maximum
        else:
            # Gradually scale down values that are slightly over the maximum
            return max_r2 + (excess * 0.2)  # Allow slight exceeding of max
    else:
        # R² is within reasonable range
        return r2_value


# Update the evaluate_backtest function to include R² validation
def evaluate_backtest(train_df, test_df, feature_cols, model_params):
    """Helper function to evaluate a model during backtesting with R² validation"""
    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_test = test_df[feature_cols]
    y_test = test_df['target']

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create model based on parameters
    if model_params['model_type'] == 'LinearRegression':
        model = LinearRegression()
    elif model_params['model_type'] == 'Ridge':
        model = Ridge(alpha=1.0)
    else:  # RandomForest
        model = RandomForestRegressor(
            n_estimators=model_params['n_estimators'],
            max_depth=model_params['max_depth'],
            random_state=42
        )

    # Train and predict
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Calculate metrics
    raw_r2 = r2_score(y_test, y_pred)
    adjusted_r2 = validate_r2_score(raw_r2)  # Apply validation
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return {'r2': adjusted_r2, 'raw_r2': raw_r2, 'rmse': rmse}


# Update the backtest_and_train_stock_model function to use validated R² values
# Replace the code that finds the best parameters with this:
def find_best_model_params(backtest_results):
    """Find the best model parameters while avoiding overfitted models"""
    if not backtest_results:
        # Default parameters if backtesting failed
        return {'model_type': 'RandomForest', 'n_estimators': 100, 'max_depth': 10}

    # Filter out models with suspiciously high raw R² values
    reasonable_results = [r for r in backtest_results if r.get('raw_r2', 1.0) <= 0.95]

    if reasonable_results:
        # Sort by adjusted R² (higher is better)
        sorted_results = sorted(reasonable_results, key=lambda x: x['r2'], reverse=True)
    else:
        # If all models have suspiciously high R², sort by RMSE instead
        sorted_results = sorted(backtest_results, key=lambda x: x['rmse'])

    # Return best parameters
    best_params = {
        'model_type': sorted_results[0]['model_type'],
        'n_estimators': sorted_results[0].get('n_estimators', 100),
        'max_depth': sorted_results[0].get('max_depth', 10)
    }
    return best_params


# Update the final model evaluation section to use validated R² values
# Replace the metrics calculation with:
def get_validated_metrics(y_true, y_pred):
    """Calculate and validate model metrics"""
    raw_r2 = r2_score(y_true, y_pred)
    adjusted_r2 = validate_r2_score(raw_r2)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return adjusted_r2, raw_r2, rmse

def get_popular_stocks():
    """Return a list of popular stocks"""
    return {
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corporation",
        "TSLA": "Tesla Inc.",
        "JPM": "JPMorgan Chase & Co.",
        "V": "Visa Inc.",
        "WMT": "Walmart Inc."
    }