# Required imports
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from openai import OpenAI
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import faiss

# Initialize session states
if 'accepted_terms' not in st.session_state:
    st.session_state.accepted_terms = False
if 'api_key_valid' not in st.session_state:
    st.session_state.api_key_valid = False
if 'data' not in st.session_state:
    st.session_state.data = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None

# Main navigation
with st.sidebar:
    try:
        st.image('images/revenuesage.jpg', use_container_width=True)
    except:
        st.title("RevenueSage")  # Fallback to text title if image fails
    options = option_menu(
        "RevenueSage",
        ["Home", "Revenue Analysis", "Forecast", "Insights"],
        icons=['house', 'graph-up', 'calendar3', 'lightbulb']
    )

def create_revenue_dashboard(data):
    # Create summary metrics
    total_revenue = data['total_order_value'].sum()
    avg_order_value = data['total_order_value'].mean()
    total_orders = data['number_of_orders'].sum()
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Revenue", f"₱{total_revenue:,.2f}")
    with col2:
        st.metric("Average Order Value", f"₱{avg_order_value:,.2f}")
    with col3:
        st.metric("Total Orders", f"{total_orders:,}")
        
    # Revenue trends
    fig = px.line(data, x='date', y='total_order_value',
                  title='Revenue Trends Over Time')
    st.plotly_chart(fig) 

def preprocess_data(data):
    """Preprocess and validate the revenue data"""
    try:
        # Convert date to datetime
        data['date'] = pd.to_datetime(data['date'])
        
        # Calculate additional features
        data['profit_margin'] = (data['total_order_value'] - data['financial_loss']) / data['total_order_value']
        data['weather_impact'] = data['weather_condition'].map({
            'normal': 0, 'heavy_rain': 1, 'typhoon': 2
        })
        
        return data
    except Exception as e:
        st.error(f"Error preprocessing data: {str(e)}")
        return None

def generate_forecast(data, forecast_period=30):
    # Prepare features for forecasting
    features = [
        'number_of_orders', 'discount_percentage', 
        'platform_commission', 'customer_satisfaction',
        'weather_impact', 'peak_season', 'courier_reliability',
        'market_demand_index', 'profit_margin'
    ]
    
    # Create time-based features
    data['month'] = data['date'].dt.month
    data['day_of_week'] = data['date'].dt.dayofweek
    features.extend(['month', 'day_of_week'])
    
    X = data[features]
    y = data['total_order_value']
    
    # Train model with more estimators and features
    model = RandomForestRegressor(n_estimators=200, max_depth=10)
    model.fit(X, y)
    
    # Generate forecast using last period's data as baseline
    future_data = X.tail(forecast_period).copy()
    future_predictions = model.predict(future_data)
    
    return future_predictions, model.feature_importances_

def display_forecast():
    if st.session_state.data is not None:
        # Preprocess data
        processed_data = preprocess_data(st.session_state.data)
        if processed_data is None:
            return
            
        # Define features list before generating forecast
        features = [
            'number_of_orders', 'discount_percentage', 
            'platform_commission', 'customer_satisfaction',
            'weather_impact', 'peak_season', 'courier_reliability',
            'market_demand_index', 'profit_margin',
            'month', 'day_of_week'
        ]
            
        # Generate forecast and feature importance
        forecast, importance = generate_forecast(processed_data)
        
        # Display forecast chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=processed_data['date'],
            y=processed_data['total_order_value'],
            name='Historical',
            line=dict(color='#1f77b4')
        ))
        
        # Add forecast with confidence interval
        forecast_dates = pd.date_range(
            start=processed_data['date'].max() + pd.Timedelta(days=1), 
            periods=30
        )
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast,
            name='Forecast',
            line=dict(color='#ff7f0e')
        ))
        
        fig.update_layout(
            title='Revenue Forecast',
            xaxis_title='Date',
            yaxis_title='Revenue (₱)',
            hovermode='x unified'
        )
        st.plotly_chart(fig)
        
        # Display feature importance
        st.subheader("Forecast Drivers")
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        st.bar_chart(importance_df.set_index('Feature')['Importance'])

def generate_revenue_insights(context, query):
    try:
        structured_prompt = f"""
        Based on the following revenue data from Philippine e-commerce operations:

        {context}

        Analyze this data and provide insights for the following query:
        {query}

        Consider:
        1. Revenue trends and patterns
        2. Customer segment performance
        3. Platform-specific insights
        4. Seasonal effects
        5. Product category performance
        6. Pricing and discount impacts

        Provide specific metrics and actionable recommendations.
        """

        client = OpenAI(api_key=st.session_state.api_key)
        chat = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are RevenueSage, an expert in e-commerce revenue analysis and forecasting."},
                {"role": "user", "content": structured_prompt}
            ],
            temperature=0.5,
            max_tokens=1500
        )
        return chat.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating insights: {str(e)}")
        return None

def load_revenue_data():
    try:
        # Read the CSV file
        df = pd.read_csv('revenue_paragraphs.csv')
        
        # Extract numerical data from the text descriptions
        df['date'] = pd.to_datetime(df['text'].str.extract(r'(\d{2}/\d{2}/\d{4})')[0])
        df['total_order_value'] = df['text'].str.extract(r'total order value reached (\d+)').astype(float)
        df['number_of_orders'] = df['text'].str.extract(r'total of (\d+) orders').astype(float)
        df['financial_loss'] = df['text'].str.extract(r'financial loss of (\d+\.?\d*)').astype(float)
        df['weather_condition'] = df['text'].str.extract(r'weather condition during this transaction was (\w+)')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Main app logic
if options == "Home":
    st.title("Welcome to RevenueSage")
    st.markdown("""
    RevenueSage is your AI-powered revenue forecasting tool designed specifically for Philippine e-commerce businesses.
    
    ### Features:
    - Revenue trend analysis
    - AI-powered forecasting
    - Weather impact analysis
    - Platform performance comparison
    - Seasonal trend detection
    """)

elif options == "Revenue Analysis":
    st.title("Revenue Analysis")
    
    if st.session_state.data is None:
        st.session_state.data = load_revenue_data()
    
    if st.session_state.data is not None:
        create_revenue_dashboard(st.session_state.data)

elif options == "Forecast":
    st.title("Revenue Forecast")
    
    if st.session_state.data is None:
        st.session_state.data = load_revenue_data()
    
    if st.session_state.data is not None:
        display_forecast()

elif options == "Insights":
    st.title("Revenue Insights")
    
    if st.session_state.data is None:
        st.session_state.data = load_revenue_data()
    
    if st.session_state.data is not None:
        query = st.text_area("What would you like to know about your revenue data?", 
                           placeholder="Example: What are the main factors affecting our revenue during typhoon season?")
        
        if st.button("Generate Insights"):
            if query:
                context = st.session_state.data.head(10).to_string()
                insights = generate_revenue_insights(context, query)
                if insights:
                    st.markdown(insights)