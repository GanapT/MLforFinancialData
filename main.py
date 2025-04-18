import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Financial Market Prediction System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os

# Import modules
from stock_predictor import run_stock_prediction
from real import run_real_estate_prediction  # Changed from real_estate_predictor to real
from crypto_predictor import run_crypto_prediction
from commodity_predictor import run_commodity_prediction
from forex_predictor import run_forex_prediction
from glossary import create_glossary_tab

# Custom CSS for better appearance
def apply_custom_css():
    st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e8cff;
        color: white;
    }
    .stButton>button {
        background-color: #4e8cff;
        color: white;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .metric-card {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
    }
    h1, h2, h3 {
        color: #1f2937;
    }
    </style>
    """, unsafe_allow_html=True)


def main():
    apply_custom_css()

    # Header with logo and title
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("https://img.icons8.com/fluency/96/financial-analytics.png", width=80)
    with col2:
        st.title("Financial Market Prediction System")
        st.markdown("### Honors Thesis by Ganap Ashit Tewary")
        st.markdown("*Advanced forecasting across multiple asset classes with ML backtesting*")

    # Create tabs for different prediction modules
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Stock Market",
        "🏘️ Real Estate",
        "🪙 Cryptocurrency",
        "🛢️ Commodities",
        "💱 Forex",
        "📚 Glossary"
    ])

    # Stock prediction tab
    with tab1:
        run_stock_prediction()

    # Real estate tab
    with tab2:
        run_real_estate_prediction()

    # Cryptocurrency tab
    with tab3:
        run_crypto_prediction()

    # Commodity tab
    with tab4:
        run_commodity_prediction()

    # Forex tab
    with tab5:
        run_forex_prediction()

    with tab6:
        create_glossary_tab()

    # Footer
    st.markdown("---")
    st.markdown("### About This System")
    st.markdown("""
    This prediction system uses advanced machine learning techniques including:
    - Backtesting against historical data for parameter optimization
    - Recursive prediction models that account for market volatility
    - Confidence intervals that widen appropriately with forecast horizon
    - Feature importance analysis to identify key market drivers
    - AI-enhanced market insights for real estate analysis

    For more information on methodology and usage, see the documentation link in the sidebar.
    """)


if __name__ == "__main__":
    main()