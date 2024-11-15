import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from fredapi import Fred
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# API configurations (you should store these securely in production)
FRED_API_KEY = '907cda4fe6006c2d940289d7949e382b'
WEATHER_API_KEY = 'ce19eb3f10be89a476a7d099f98dcd40'
fred = Fred(api_key=FRED_API_KEY)

# Data fetching functions
def get_market_data(symbol, start_date, end_date):
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        data.index = data.index.tz_localize(None)
        return data['Close']
    except Exception as e:
        st.error(f"Error fetching market data: {e}")
        return None

def get_economic_data(series_id, start_date, end_date):
    try:
        return fred.get_series(series_id, start_date, end_date)
    except Exception as e:
        st.error(f"Error fetching economic data: {e}")
        return None

def get_weather_data(city):
    try:
        base_url = "http://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": city,
            "appid": WEATHER_API_KEY
        }
        response = requests.get(base_url, params=params)
        return response.json()
    except Exception as e:
        st.error(f"Error fetching weather data: {e}")
        return None

# Data source mappings
market_symbols = {
    "VIX Index": "^VIX",
    "Gold Prices": "GC=F",
    "Oil Prices": "CL=F",
    "Bitcoin Price": "BTC-USD",
    "Dollar Index": "DX-Y.NYB"
}

economic_series = {
    "US Interest Rates": "FEDFUNDS",
    "Inflation Rate": "CPIAUCSL",
    "Unemployment Rate": "UNRATE",
    "GDP Growth": "GDP"
}

# Define some sample factors
# Define factors
factors = {
    "Economic Indicators": {
        "US Interest Rates": "Federal Reserve interest rate decisions and market impact",
        "Inflation Rate": "CPI and its effect on markets",
        "GDP Growth": "Economic growth indicators",
        "Unemployment Rate": "Labor market health",
        "Consumer Confidence": "Consumer sentiment index"
    },
    "Market Metrics": {
        "VIX Index": "Market volatility indicator",
        "Dollar Index": "USD strength against major currencies",
        "Gold Prices": "Safe haven asset movements",
        "Oil Prices": "Energy market dynamics",
        "Bitcoin Price": "Crypto market benchmark"
    },
    "Religious Events": {
        "Ramadan Timing": "Islamic calendar dates and their potential impact on consumer behavior",
        "Christmas Season": "Western holiday season impact on retail and markets",
        "Diwali": "Indian festival impact on gold prices and local markets",
        "Lunar New Year": "Asian markets seasonal patterns",
        "Easter": "Spring retail season impact"
    },
    "Environmental Factors": {
        "River Water Temperature": "Impact on nuclear power plant output",
        "Saudi Arabia Climate": "Effect on global gas prices",
        "El Niño Events": "Impact on agricultural commodities",
        "Hurricane Season": "Effect on insurance and energy sectors",
        "Drought Conditions": "Agricultural yield impacts",
        "Air Quality Index": "Environmental regulation impacts"
    },
    "Social Factors": {
        "Major Sports Events": "Impact on advertising spending and media stocks",
        "Political Elections": "Market uncertainty and sector-specific impacts",
        "Social Media Trends": "Consumer sentiment and retail trading patterns",
        "Academic Calendar": "Seasonal education sector impacts",
        "Tourism Season": "Travel and hospitality sector patterns"
    },
    "Technological Trends": {
        "Semiconductor Demand": "Tech sector leading indicator",
        "Cloud Computing Usage": "Enterprise tech adoption metrics",
        "5G Rollout": "Telecom infrastructure development",
        "AI Model Releases": "Tech innovation cycles",
        "Cybersecurity Incidents": "Digital risk metrics"
    }
}

# Add this class for technical analysis and predictions
class MarketAnalyzer:
    def __init__(self, df):
        self.df = df.copy()
        # Rename columns to match expected format
        if selected_factor1 in market_symbols:
            self.df.rename(columns={selected_factor1: 'Close'}, inplace=True)
        elif selected_factor2 in market_symbols:
            self.df.rename(columns={selected_factor2: 'Close'}, inplace=True)
        self.model = None

    def create_features(self):
        if 'Close' not in self.df.columns:
            st.error("No market price data available for technical analysis")
            return self.df
            
        # Technical indicators
        self.df['SMA_20'] = self.df['Close'].rolling(window=20).mean()
        self.df['SMA_50'] = self.df['Close'].rolling(window=50).mean()
        self.df['Returns'] = self.df['Close'].pct_change()
        self.df['Volatility'] = self.df['Returns'].rolling(window=20).std()
        
        return self.df.dropna()

    def build_prediction_model(self, features):
        df = self.create_features()
        
        X = df[features]
        y = df['Returns']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        
        return {
            'mse': mean_squared_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'feature_importance': dict(zip(features, self.model.coef_))
        }

# Set page config
st.set_page_config(page_title="Market Factor Analysis", layout="wide")

# Add sidebar navigation
page = st.sidebar.selectbox(
    'Select Analysis Page',
    ['Correlation Analysis', 'Technical Analysis', 'Model Predictions']
)

# Sidebar for factor selection and date range
st.sidebar.header("Select Factors to Compare")

# Date range selection
start_date = st.sidebar.date_input(
    "Start Date",
    datetime.now() - timedelta(days=365)
)
end_date = st.sidebar.date_input(
    "End Date",
    datetime.now()
)

# Random factor suggestion
if 'factor1' not in st.session_state:
    categories = list(factors.keys())
    random_category1 = np.random.choice(categories)
    random_factor1 = np.random.choice(list(factors[random_category1].keys()))
    random_category2 = np.random.choice(categories)
    random_factor2 = np.random.choice(list(factors[random_category2].keys()))
    st.session_state.factor1 = (random_category1, random_factor1)
    st.session_state.factor2 = (random_category2, random_factor2)

# Factor selection
selected_category1 = st.sidebar.selectbox(
    "Select Category 1",
    options=factors.keys(),
    index=list(factors.keys()).index(st.session_state.factor1[0])
)

# Update session state if category changes
if selected_category1 != st.session_state.factor1[0]:
    st.session_state.factor1 = (selected_category1, list(factors[selected_category1].keys())[0])

selected_factor1 = st.sidebar.selectbox(
    "Select Factor 1",
    options=factors[selected_category1].keys(),
    index=list(factors[selected_category1].keys()).index(st.session_state.factor1[1])
)

selected_category2 = st.sidebar.selectbox(
    "Select Category 2",
    options=factors.keys(),
    index=list(factors.keys()).index(st.session_state.factor2[0])
)

# Update session state if category changes
if selected_category2 != st.session_state.factor2[0]:
    st.session_state.factor2 = (selected_category2, list(factors[selected_category2].keys())[0])

selected_factor2 = st.sidebar.selectbox(
    "Select Factor 2",
    options=factors[selected_category2].keys(),
    index=list(factors[selected_category2].keys()).index(st.session_state.factor2[1])
)

# Update session state with selected factors
st.session_state.factor1 = (selected_category1, selected_factor1)
st.session_state.factor2 = (selected_category2, selected_factor2)

# Display selected factors
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Factor 1: {selected_factor1}")
    st.write(factors[selected_category1][selected_factor1])

with col2:
    st.subheader(f"Factor 2: {selected_factor2}")
    st.write(factors[selected_category2][selected_factor2])

# Fetch real data if available, otherwise use simulated data
def get_factor_data(factor_name, start_date, end_date):
    if factor_name in market_symbols:
        return get_market_data(market_symbols[factor_name], start_date, end_date)
    elif factor_name in economic_series:
        return get_economic_data(economic_series[factor_name], start_date, end_date)
    else:
        # Generate simulated data for factors without real data source
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        dates = dates.tz_localize(None)
        return pd.Series(
            np.sin(np.arange(len(dates)) * 2 * np.pi / 365) + np.random.normal(0, 0.2, len(dates)),
            index=dates
        )

# Get data for both factors
factor1_data = get_factor_data(selected_factor1, start_date, end_date)
factor2_data = get_factor_data(selected_factor2, start_date, end_date)

# Create DataFrame with aligned dates
df = pd.DataFrame({
    selected_factor1: factor1_data,
    selected_factor2: factor2_data
})
df = df.dropna()  # Remove any missing values

if page == 'Correlation Analysis':
    if len(df) > 0:
        # Correlation analysis
        correlation = df[selected_factor1].corr(df[selected_factor2])

        # Visualization with dual y-axes
        st.subheader("Time Series Comparison")
        
        # Create figure with secondary y-axis
        fig = go.Figure()

        # Normalize data to percentage change from first value
        norm_factor1 = ((df[selected_factor1] / df[selected_factor1].iloc[0]) - 1) * 100
        norm_factor2 = ((df[selected_factor2] / df[selected_factor2].iloc[0]) - 1) * 100

        # Add traces
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=norm_factor1,
                name=f"{selected_factor1} (% change)",
                line=dict(color='blue')
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=norm_factor2,
                name=f"{selected_factor2} (% change)",
                line=dict(color='red')
            )
        )

        # Update layout
        fig.update_layout(
            title="Factor Comparison Over Time (Normalized to % Change)",
            xaxis_title="Date",
            yaxis_title="Percentage Change (%)",
            height=500,
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        # Add range slider
        fig.update_xaxes(rangeslider_visible=True)

        st.plotly_chart(fig, use_container_width=True)

        # Add raw data toggle
        if st.checkbox("Show raw values"):
            fig_raw = go.Figure()
            
            # Add traces with secondary y-axis
            fig_raw.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[selected_factor1],
                    name=selected_factor1,
                    line=dict(color='blue')
                )
            )

            fig_raw.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[selected_factor2],
                    name=selected_factor2,
                    yaxis="y2",
                    line=dict(color='red')
                )
            )

            # Update layout with secondary y-axis
            fig_raw.update_layout(
                title="Raw Values Comparison (Dual Axis)",
                xaxis_title="Date",
                yaxis_title=f"{selected_factor1} Value",
                yaxis2=dict(
                    title=f"{selected_factor2} Value",
                    overlaying="y",
                    side="right"
                ),
                height=500,
                hovermode='x unified',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )

            st.plotly_chart(fig_raw, use_container_width=True)

        # Scatter plot
        st.subheader("Correlation Analysis")
        fig_scatter = px.scatter(
            df, 
            x=selected_factor1, 
            y=selected_factor2, 
            trendline="ols",
            title=f"Correlation: {correlation:.2f}"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Additional insights
        st.subheader("Analysis Insights")
        st.write(f"""
        - The correlation coefficient between {selected_factor1} and {selected_factor2} is {correlation:.2f}
        - This indicates a {'strong' if abs(correlation) > 0.7 else 'moderate' if abs(correlation) > 0.3 else 'weak'} 
          {'positive' if correlation > 0 else 'negative'} correlation
        - The visualization shows {'clear' if abs(correlation) > 0.5 else 'some'} patterns in the relationship between these factors
        """)
    else:
        st.error("No data available for the selected date range and factors")

elif page == 'Technical Analysis':
    if len(df) > 0:
        # Check if either factor is a market symbol
        if selected_factor1 in market_symbols or selected_factor2 in market_symbols:
            analyzer = MarketAnalyzer(df)
            tech_df = analyzer.create_features()
            
            if 'Close' in tech_df.columns:
                st.subheader('Technical Indicators')
                
                # Technical analysis plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=tech_df.index, y=tech_df['Close'], name='Price'))
                fig.add_trace(go.Scatter(x=tech_df.index, y=tech_df['SMA_20'], name='20-day SMA'))
                fig.add_trace(go.Scatter(x=tech_df.index, y=tech_df['SMA_50'], name='50-day SMA'))
                
                fig.update_layout(
                    title="Price and Moving Averages",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Volatility plot
                st.subheader('Price Volatility')
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Scatter(x=tech_df.index, y=tech_df['Volatility'], name='Volatility'))
                fig_vol.update_layout(
                    title="20-day Rolling Volatility",
                    xaxis_title="Date",
                    yaxis_title="Volatility",
                    height=400
                )
                st.plotly_chart(fig_vol, use_container_width=True)
        else:
            st.warning("Technical analysis is only available for market price data. Please select a market factor (VIX, Gold, Oil, Bitcoin, or Dollar Index).")

elif page == 'Model Predictions':
    if len(df) > 0:
        # Check if either factor is a market symbol
        if selected_factor1 in market_symbols or selected_factor2 in market_symbols:
            st.subheader('Prediction Model')
            
            analyzer = MarketAnalyzer(df)
            tech_df = analyzer.create_features()
            
            if 'Close' in tech_df.columns:
                # Select features for the model
                features = st.multiselect(
                    'Select features for the model',
                    options=['SMA_20', 'SMA_50', 'Volatility'],
                    default=['SMA_20', 'SMA_50', 'Volatility']
                )
                
                if features:
                    model_results = analyzer.build_prediction_model(features)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("R-squared Score", f"{model_results['r2']:.4f}")
                    with col2:
                        st.metric("Mean Squared Error", f"{model_results['mse']:.6f}")
                    
                    # Feature importance plot
                    st.subheader('Feature Importance')
                    importance_df = pd.DataFrame.from_dict(
                        model_results['feature_importance'], 
                        orient='index', 
                        columns=['Importance']
                    ).sort_values('Importance', ascending=False)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=importance_df.index,
                        y=importance_df['Importance'],
                        name='Feature Importance'
                    ))
                    fig.update_layout(
                        title="Feature Importance in Prediction Model",
                        xaxis_title="Features",
                        yaxis_title="Importance",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Prediction model is only available for market price data. Please select a market factor (VIX, Gold, Oil, Bitcoin, or Dollar Index).")

# Future enhancements section
st.sidebar.markdown("""
---
### Future Enhancements
- Additional data sources integration
- Advanced statistical analysis
- Custom factor creation
- Machine learning predictions
- Alert system for pattern changes
""")
