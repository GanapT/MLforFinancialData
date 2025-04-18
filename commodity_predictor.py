import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests

# Polygon.io API key
POLYGON_API_KEY = "WNvXC2TTYZah3EFclB3w2nyCaKOKzk3R"


def get_commodity_data(ticker, timespan="day", limit=365):
    """Get commodity ETF data from Polygon.io"""
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
        st.error(f"Error retrieving commodity data: {e}")
        return None


def prepare_commodity_data(df):
    """Prepare commodity data for modeling"""
    # Create target variable
    df['target'] = df['Close']

    # Create date features
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['day_of_month'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['quarter'] = df['Date'].dt.quarter

    # Create time series features
    df['time_idx'] = range(len(df))

    # Create technical indicators
    # Moving Averages
    for window in [5, 10, 20, 50]:
        if len(df) > window:
            df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()

    # Volatility
    for window in [10, 20, 50]:
        if len(df) > window:
            df[f'Volatility_{window}'] = df['Close'].rolling(window=window).std()

    # RSI
    if len(df) > 14:
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

    # Price momentum
    for window in [1, 5, 10, 20]:
        if len(df) > window:
            df[f'Return_{window}d'] = df['Close'].pct_change(window)

    # Volume indicators
    if 'Volume' in df.columns:
        for window in [5, 10, 20]:
            if len(df) > window:
                df[f'Volume_MA_{window}'] = df['Volume'].rolling(window=window).mean()
                df[f'Volume_Ratio_{window}'] = df['Volume'] / df[f'Volume_MA_{window}']

    # Drop NaN values
    df.dropna(inplace=True)

    return df


def backtest_commodity_model(df, future_days=30, test_periods=5):
    """Train model with backtesting to improve prediction accuracy"""
    # Prepare data
    df_processed = prepare_commodity_data(df.copy())

    # Features for prediction
    feature_cols = [col for col in df_processed.columns if
                    col not in ['Date', 'timestamp', 'target', 'Open', 'High', 'Low', 'Close', 'Volume']]

    # Backtesting on multiple historical periods to optimize the model
    backtest_results = []

    # Try different time periods for backtesting
    for i in range(test_periods):
        # Define test period - sliding window approach
        test_start = len(df_processed) - (i + 1) * 30
        test_end = len(df_processed) - i * 30

        if test_start < 60:  # Ensure enough data for training
            continue

        # Split data
        backtest_train = df_processed.iloc[:test_start].copy()
        backtest_test = df_processed.iloc[test_start:test_end].copy()

        # Try different model parameters
        for n_est in [50, 100, 200]:
            for lr in [0.05, 0.1, 0.2]:
                for depth in [3, 5, 7]:
                    # Train model with current parameters
                    X_train = backtest_train[feature_cols]
                    y_train = backtest_train['target']
                    X_test = backtest_test[feature_cols]
                    y_test = backtest_test['target']

                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    # Train model
                    model = GradientBoostingRegressor(
                        n_estimators=n_est,
                        learning_rate=lr,
                        max_depth=depth,
                        random_state=42
                    )
                    model.fit(X_train_scaled, y_train)

                    # Make predictions
                    y_pred = model.predict(X_test_scaled)

                    # Calculate metrics
                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                    # Store results
                    backtest_results.append({
                        'n_estimators': n_est,
                        'learning_rate': lr,
                        'max_depth': depth,
                        'r2': r2,
                        'rmse': rmse
                    })

    # Find best parameters based on backtesting
    if backtest_results:
        # Sort by R² score (higher is better)
        sorted_results = sorted(backtest_results, key=lambda x: x['r2'], reverse=True)
        best_params = {
            'n_estimators': sorted_results[0]['n_estimators'],
            'learning_rate': sorted_results[0]['learning_rate'],
            'max_depth': sorted_results[0]['max_depth']
        }
    else:
        # Default parameters if backtesting failed
        best_params = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3}

    # Train final model with best parameters on all training data
    X = df_processed[feature_cols]
    y = df_processed['target']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train final model with best parameters
    model = GradientBoostingRegressor(
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth'],
        random_state=42
    )
    model.fit(X_scaled, y)

    # Make predictions on entire dataset to assess fit
    train_predictions = model.predict(X_scaled)

    # Calculate final metrics
    r2 = r2_score(y, train_predictions)
    rmse = np.sqrt(mean_squared_error(y, train_predictions))

    # Generate future dates for prediction
    last_date = df['Date'].iloc[-1]
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=future_days)

    # Recursive prediction approach for more realistic predictions
    # Start with recent historical window
    prediction_window = df.copy().tail(50)
    future_values = []

    # Walk-forward prediction for each future day
    for i in range(future_days):
        # Get current date
        current_date = future_dates[i]

        # Process the current window
        processed_window = prepare_commodity_data(prediction_window.copy())

        # Create a new row with all required features
        if len(processed_window) > 0:
            # First get whatever features we can from the processed window
            last_row_data = processed_window.iloc[-1:].to_dict('records')[0]

            # Create a new DataFrame with exactly the same features and order as used in training
            last_features = pd.DataFrame(columns=feature_cols)
            last_features.loc[0] = np.nan  # Initialize with NaN

            # Fill in available values from the processed window
            for col in feature_cols:
                if col in last_row_data:
                    last_features.loc[0, col] = last_row_data[col]
                else:
                    # Use the mean from training data for missing features
                    last_features.loc[0, col] = X[col].mean()

            # Now we have a properly formatted DataFrame with all required features
            last_features_scaled = scaler.transform(last_features)

            # Predict next price
            next_price = model.predict(last_features_scaled)[0]

            # Add realistic volatility
            volatility = df['Close'].pct_change().std()
            random_factor = np.random.normal(0, volatility)
            next_price = next_price * (1 + random_factor * 0.3)
        else:
            # Fallback if processing fails
            last_price = prediction_window['Close'].iloc[-1]
            volatility = df['Close'].pct_change().std() if len(df) > 1 else 0.01
            next_price = last_price * (1 + np.random.normal(0, volatility))

        # Store prediction
        future_values.append(next_price)

        # Create new row for next day's window
        new_row = pd.DataFrame({
            'Date': [current_date],
            'Open': [prediction_window['Close'].iloc[-1]],
            'High': [
                max(prediction_window['Close'].iloc[-1], next_price) * (1 + np.random.normal(0, volatility * 0.5))],
            'Low': [
                min(prediction_window['Close'].iloc[-1], next_price) * (1 - np.random.normal(0, volatility * 0.5))],
            'Close': [next_price],
            'Volume': [prediction_window['Volume'].mean()]
        })

        # Update window for next prediction
        prediction_window = pd.concat([prediction_window.iloc[1:], new_row], ignore_index=True)

    # Create future predictions dataframe
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Price': future_values
    })

    # Calculate confidence intervals that widen over time
    daily_volatility = df['Close'].pct_change().std()
    time_factor = np.sqrt(np.arange(1, len(future_df) + 1) / 5)

    future_df['Upper_Bound'] = future_df['Predicted_Price'] * (1 + 1.96 * daily_volatility * time_factor)
    future_df['Lower_Bound'] = future_df['Predicted_Price'] * (1 - 1.96 * daily_volatility * time_factor)

    # Calculate feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    # Enhanced results object
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


def train_commodity_model(df, future_days=30):
    """Train a commodity prediction model with backtesting"""
    return backtest_commodity_model(df, future_days)


def create_commodity_chart(df, name):
    """Create an interactive chart for commodity data"""
    # Create figure with secondary y-axis for volume
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        specs=[[{"type": "candlestick"}], [{"type": "bar"}]]
    )

    # Add price candlestick chart
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
    for window, color in zip([20, 50, 200], ['blue', 'orange', 'purple']):
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

    # Add volume bars with conditional coloring
    colors = ['green' if df['Close'].iloc[i] >= df['Open'].iloc[i] else 'red'
              for i in range(len(df))]

    fig.add_trace(
        go.Bar(
            x=df['Date'],
            y=df['Volume'],
            name="Volume",
            marker_color=colors
        ),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        title=f"{name} Historical Price",
        xaxis_title="Date",
        xaxis_rangeslider_visible=False,
        yaxis_title="Price ($)",
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

    # Update axes labels
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig


def create_prediction_chart(df, results, name):
    """Create an interactive chart for commodity predictions"""
    fig = go.Figure()

    # Add historical data
    fig.add_trace(
        go.Scatter(
            x=df['Date'].iloc[-90:],  # Show last 90 days for clearer visualization
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
        title=f"{name} Price Prediction",
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


def get_commodity_symbols():
    """Return a list of commodities with verified ETF tickers that work with Polygon.io API"""
    return {
        "SLV": "Silver ETF (iShares)",
        "USO": "Crude Oil ETF",
        "UNG": "Natural Gas ETF",
        "CPER": "Copper ETF",
        "CORN": "Corn ETF",
        "SOYB": "Soybean ETF",
        "WEAT": "Wheat ETF",
        "JO": "Coffee ETF",
        "SGG": "Sugar ETF",
        "NIB": "Cocoa ETF",
        "BAL": "Cotton ETF"
    }


def run_commodity_prediction():
    """Main function for commodity prediction module"""
    st.header("Commodity Price Prediction")

    # Sidebar controls
    st.sidebar.subheader("Commodity Parameters")
    st.sidebar.markdown("#### Step 1: Select Commodity")

    # Get commodity symbols
    commodity_symbols = get_commodity_symbols()

    # Create formatted options for the selectbox
    options = [f"{name} ({symbol})" for symbol, name in commodity_symbols.items()]
    selected_option = st.sidebar.selectbox("Select Commodity", options, key="commodity_select")

    # Extract symbol from the selected option using rsplit for reliability
    selected_symbol = selected_option.rsplit('(', 1)[-1].replace(')', '').strip()

    # Advanced options
    with st.sidebar.expander("Advanced Options", expanded=False):
        timespan = st.selectbox(
            "Data Frequency",
            ["day", "week", "month"],
            index=0,
            key="commodity_timespan"
        )
        future_days = st.slider(
            "Days to Predict",
            5, 60, 30,
            key="commodity_future_days"
        )
        test_periods = st.slider(
            "Backtest Periods",
            1, 10, 5,
            key="commodity_test_periods"
        )

    # Initialize session state
    if 'commodity_data' not in st.session_state:
        st.session_state.commodity_data = None
    if 'commodity_results' not in st.session_state:
        st.session_state.commodity_results = None

    # Market overview information
    with st.expander("Commodity Market Insights", expanded=False):
        st.markdown("""
        ### Key Commodity Market Factors

        When analyzing commodity markets, these factors significantly impact price movements:

        1. **Supply and Demand Fundamentals**: The most basic driver of commodity prices is the balance between 
           production capacity and consumption needs. Supply disruptions (weather, geopolitical issues) or demand 
           shocks can cause significant price volatility.

        2. **Seasonal Patterns**: Many commodities follow seasonal patterns based on production cycles, 
           particularly agricultural products like corn, wheat, and soybeans.

        3. **Macroeconomic Influences**: Inflation expectations, interest rate changes, and currency fluctuations 
           (especially the US dollar) strongly impact commodity prices.

        4. **Inventory Levels**: Current stockpiles relative to demand create price pressure - low inventories 
           typically support higher prices while abundant supplies tend to depress prices.

        Our model incorporates technical indicators and historical patterns to predict future price movements, 
        but cannot account for unexpected events like natural disasters or major policy changes.
        """)

    # Load data button
    col1, col2 = st.columns([3, 1])
    with col2:
        load_data = st.button("Load Commodity Data", key="commodity_load_btn", use_container_width=True)

    if load_data:
        with st.spinner(f"Loading data for {selected_symbol}..."):
            df = get_commodity_data(selected_symbol, timespan)
            if df is not None and not df.empty:
                st.session_state.commodity_data = df
                st.session_state.commodity_symbol = selected_symbol
                st.session_state.commodity_name = commodity_symbols[selected_symbol]
                st.success(f"Successfully loaded data for {commodity_symbols[selected_symbol]}")

                # Display price chart immediately after loading
                chart = create_commodity_chart(df, commodity_symbols[selected_symbol])
                st.plotly_chart(chart, use_container_width=True)

                # Show basic statistics
                with st.expander("Market Statistics", expanded=True):
                    # Calculate key metrics
                    current_price = df['Close'].iloc[-1]
                    prev_price = df['Close'].iloc[-2]
                    daily_change = (current_price - prev_price) / prev_price * 100

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Current Price", f"${current_price:.2f}", f"{daily_change:.2f}%")

                    # 52-week high/low
                    high_52w = df['High'].iloc[-252:].max() if len(df) >= 252 else df['High'].max()
                    low_52w = df['Low'].iloc[-252:].min() if len(df) >= 252 else df['Low'].min()
                    col2.metric("52-Week High", f"${high_52w:.2f}")
                    col3.metric("52-Week Low", f"${low_52w:.2f}")

                    # Average volume
                    avg_vol = df['Volume'].mean()
                    col4.metric("Avg. Volume", f"{avg_vol:,.0f}")

                    # Historical performance
                    st.markdown("---")
                    col1, col2 = st.columns(2)

                    with col1:
                        # Calculate returns
                        monthly_return = (current_price / df['Close'].iloc[-22] - 1) * 100 if len(df) >= 22 else None
                        quarterly_return = (current_price / df['Close'].iloc[-66] - 1) * 100 if len(df) >= 66 else None
                        yearly_return = (current_price / df['Close'].iloc[-252] - 1) * 100 if len(df) >= 252 else None

                        st.markdown("#### Performance")
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
                        # Calculate volatility and risk metrics
                        daily_vol = df['Close'].pct_change().std() * 100

                        # Calculate Sharpe ratio if we have yearly data
                        sharpe = None
                        if yearly_return is not None:
                            risk_free_rate = 3.0  # Assuming 3% risk-free rate
                            sharpe = (yearly_return - risk_free_rate) / (daily_vol * np.sqrt(252))

                        st.markdown("#### Risk Metrics")
                        vol_data = pd.DataFrame({
                            'Metric': ['Daily Volatility', 'Annualized Volatility', 'Sharpe Ratio'],
                            'Value': [
                                f"{daily_vol:.2f}%",
                                f"{daily_vol * np.sqrt(252):.2f}%",
                                f"{sharpe:.2f}" if sharpe is not None else "N/A"
                            ]
                        })
                        st.dataframe(vol_data, hide_index=True, use_container_width=True)
            else:
                st.error(f"Could not retrieve data for {selected_symbol}. Please try another commodity.")

    # Display data and run predictions if data is loaded
    if st.session_state.commodity_data is not None:
        df = st.session_state.commodity_data
        symbol = st.session_state.commodity_symbol
        name = st.session_state.commodity_name

        # Train model button
        train_model = st.button("Generate Price Predictions", key="commodity_predict_btn", use_container_width=True)

        if train_model:
            with st.spinner("Training model and generating predictions..."):
                results = train_commodity_model(df.copy(), future_days=future_days)
                st.session_state.commodity_results = results
                st.success("Predictions generated successfully!")

        # Display predictions if available
        if st.session_state.commodity_results is not None:
            results = st.session_state.commodity_results

            # Display prediction chart
            st.markdown("### Price Predictions")
            pred_chart = create_prediction_chart(df, results, name)
            st.plotly_chart(pred_chart, use_container_width=True)

            # Display backtesting results
            with st.expander("Backtesting Analysis", expanded=False):
                st.markdown("#### Model Optimization Process")
                st.markdown(f"""
                The model was optimized by testing {len(results['backtest_results'])} different 
                configurations on historical data. The optimal parameters found were:

                - n_estimators: {results['best_params']['n_estimators']}
                - learning_rate: {results['best_params']['learning_rate']}
                - max_depth: {results['best_params']['max_depth']}
                """)

                # Show parameter impact on performance
                if len(results['backtest_results']) > 5:
                    backtest_df = pd.DataFrame(results['backtest_results'])

                    # Group by n_estimators
                    param_impact = backtest_df.groupby('n_estimators')['r2'].mean().reset_index()

                    # Create bar chart
                    param_fig = go.Figure(go.Bar(
                        x=param_impact['n_estimators'].astype(str),
                        y=param_impact['r2'],
                        marker_color='#4e8cff'
                    ))

                    param_fig.update_layout(
                        title="Effect of Tree Count on Model Performance",
                        xaxis_title="Number of Estimators",
                        yaxis_title="Average R² Score",
                        height=300,
                        template="plotly_white"
                    )

                    st.plotly_chart(param_fig, use_container_width=True)

            # Display feature importance
            with st.expander("Price Drivers Analysis", expanded=False):
                st.markdown("#### Key Factors Influencing Predictions")

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

                # Provide market context
                most_important = top_features['Feature'].iloc[0]
                second_important = top_features['Feature'].iloc[1]

                st.markdown(f"""
                ### Market Analysis

                Based on the feature importance analysis, the model identifies **{most_important}** and 
                **{second_important}** as the most significant factors for predicting {name} prices.

                This suggests that:

                - {"Recent price trends are highly influential for future price movements" if "lag" in most_important or "MA" in most_important else "Technical indicators are driving price action more than seasonality"}
                - {"Price momentum is a key driver" if "Return" in most_important or "Return" in second_important else "Market volatility is playing a significant role" if "Volatility" in most_important or "Volatility" in second_important else "Recent price levels are more important than longer-term trends"}

                Commodity traders should focus on these factors when making investment decisions.
                """)

            # Display price prediction table
            with st.expander("Detailed Price Predictions", expanded=True):
                future_df = results['future_predictions']

                # Format for display
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
                    file_name=f"{symbol}_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
    else:
        # Display placeholder if no data is loaded
        st.info("Please load commodity data using the sidebar controls")