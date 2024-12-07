# Required imports for application functionality
import os
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import faiss
from openai import OpenAI
import tiktoken
from langchain_community.llms import OpenAI as LangChainOpenAI

# Configure Streamlit page settings - MUST BE FIRST!
st.set_page_config(page_title="ReveNEW", page_icon="", layout="wide")

# Initialize session state variables
if 'accepted_terms' not in st.session_state:
    st.session_state.accepted_terms = False
if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False
if 'embeddings_created' not in st.session_state:
    st.session_state.embeddings_created = False
if 'index_ready' not in st.session_state:
    st.session_state.index_ready = False
if 'api_key_valid' not in st.session_state:
    st.session_state.api_key_valid = False
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
if 'nlg_template' not in st.session_state:
    st.session_state.nlg_template = None

# Display warning page for first-time users
if not st.session_state.accepted_terms:
    st.markdown("""
        <style>
        .warning-header {
            color: white;
            text-align: center;
            padding: 20px;
            margin-bottom: 20px;
        }
        .warning-section {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 15px;
            border-left: 4px solid #ff4b4b;
            color: black;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.markdown("<h1 class='warning-header'>Welcome to ReveNEW</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='warning-section'>", unsafe_allow_html=True)
    st.markdown("### 1. Data Security")
    st.markdown("""
    - Your financial data remains confidential
    - Secure API key handling
    - Local data processing when possible
    - Regular security updates
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    agree = st.checkbox("I understand and agree to the data handling policies")
    if st.button("Continue to ReveNEW", disabled=not agree):
        st.session_state.accepted_terms = True
        st.rerun()
    st.stop()

# Function to process revenue data and create embeddings
def process_revenue_data(text_data):
    try:
        client = OpenAI(api_key=st.session_state.api_key)
        response = client.embeddings.create(
            input=text_data,
            model="text-embedding-ada-002"
        )
        return np.array([response.data[0].embedding])
    except Exception as e:
        st.error(f"Error creating embeddings: {str(e)}")
        return None

# Function to generate NLG template
def generate_nlg_template(df_info):
    try:
        client = OpenAI(api_key=st.session_state.api_key)
        prompt = f"""
        Create a natural language template for revenue analysis with these data points:
        {df_info}
        Focus on revenue trends, forecasting insights, and business opportunities.
        Use clear business language and include specific metrics.
        """
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a financial analyst creating revenue forecast reports."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating template: {str(e)}")
        return None

# Function to generate revenue analysis
def generate_revenue_analysis(context, query):
    try:
        structured_prompt = f"""
        Based on the following revenue data:

        {context}

        Please analyze this data as ReveNEW and answer the following query:
        {query}

        Provide specific insights considering:
        1. Revenue trends and patterns
        2. Growth opportunities
        3. Seasonal factors
        4. Market conditions
        5. Customer segments
        """

        client = OpenAI(api_key=st.session_state.api_key)
        chat = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are ReveNEW, an AI expert in revenue forecasting and analysis."},
                {"role": "user", "content": structured_prompt}
            ],
            temperature=0.5,
            max_tokens=1500
        )
        return chat.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating analysis: {str(e)}")
        return None

# Sidebar setup
with st.sidebar:
    st.image('images/revenew.jpg', use_container_width=True)
    
    # Move navigation menu to sidebar
    options = option_menu(
        menu_title="Navigation",
        options=["Home", "Revenue Analysis"],
        icons=["house", "graph-up"],
        menu_icon="cash-coin",
        default_index=0,
        styles={
            "container": {"padding": "0!important"},
            "icon": {"color": "#FFD700", "font-size": "20px"},
            "nav-link": {"font-size": "17px", "text-align": "left", "margin": "5px"},
            "nav-link-selected": {"background-color": "#2E2E2E"}
        }
    )
    
    # Add some spacing
    st.markdown("---")
    
    # API Key section
    st.markdown('<p style="color: white;">OpenAI API Key:</p>', unsafe_allow_html=True)
    col1, col2 = st.columns([5,1], gap="small")
    with col1:
        api_key = st.text_input('', type='password', label_visibility="collapsed")
    with col2:
        check_api = st.button('>', key='api_button')
    
    if check_api:
        if not api_key:
            st.warning('Please enter your OpenAI API token!')
        else:
            try:
                client = OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=5
                )
                st.session_state.api_key = api_key
                st.session_state.api_key_valid = True
                st.success('API key is valid!')
            except Exception as e:
                st.error('Invalid API key or API error occurred')
                st.session_state.api_key_valid = False

# Options: Home
if options == "Home":
    st.markdown("<h1 style='text-align: center; margin-bottom: 15px; color: white;'>Welcome to ReveNEW!</h1>", unsafe_allow_html=True)
    
    st.markdown("<div style='text-align: center; padding: 10px; margin-bottom: 20px; font-size: 18px; color: white;'>ReveNEW is your intelligent companion for revenue forecasting and identifying new monetization opportunities. Our AI-powered system analyzes historical sales data, market trends, and user behavior to provide accurate forecasts and insights.</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<h3 style='text-align: center; color: #FFD700; margin-bottom: 10px;'>Key Features</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; font-size: 16px; color: black; min-height: 200px;'>
        <ul style='list-style-type: none; padding-left: 0; margin: 0;'>
        <li style='margin-bottom: 8px;'>• Revenue Forecasting</li>
        <li style='margin-bottom: 8px;'>• Market Trend Analysis</li>
        <li style='margin-bottom: 8px;'>• Customer Segmentation Insights</li>
        <li style='margin-bottom: 8px;'>• Growth Opportunity Identification</li>
        <li style='margin-bottom: 8px;'>• Seasonal Impact Analysis</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# Options: Revenue Analysis
elif options == "Revenue Analysis":
    st.title("Revenue Analysis")
    
    # File uploader for revenue data
    uploaded_file = st.file_uploader("Upload your revenue data (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Process the uploaded file
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.data_processed = True
            
            # Convert dataframe to text for processing
            text_data = df.to_string()
            
            # Create embeddings
            embeddings = process_revenue_data(text_data)
            
            if embeddings is not None:
                st.session_state.embeddings = embeddings
                st.session_state.documents = [text_data]
                
                # Create FAISS index
                dimension = embeddings.shape[1]
                index = faiss.IndexFlatL2(dimension)
                index.add(embeddings)
                st.session_state.faiss_index = index
                
                st.success("Data processed successfully!")
                
                st.markdown("### Ask ReveNEW Questions")
                
                # Single query input with dropdown for focus area
                query_type = st.selectbox(
                    "Select Analysis Focus:",
                    [
                        "Revenue Trends",
                        "Market Opportunities",
                        "Customer Segmentation",
                        "Seasonal Impact",
                        "Comprehensive Analysis"
                    ]
                )
                
                # Dynamic placeholder based on selection
                placeholders = {
                    "Revenue Trends": "Example: What are the revenue trends for the last quarter?",
                    "Market Opportunities": "Example: What new monetization opportunities exist in our market?",
                    "Customer Segmentation": "Example: Which customer segments are driving our growth?",
                    "Seasonal Impact": "Example: How do seasonal factors affect our revenue?",
                    "Comprehensive Analysis": "Example: Analyze our overall revenue performance and opportunities"
                }
                
                query = st.text_area(
                    "Your Question:",
                    placeholder=placeholders[query_type],
                    help="Be specific with timeframes, sectors, or metrics for better results",
                    height=100
                )
                
                if st.button("Generate Analysis"):
                    if query.strip():
                        # Get similar documents
                        query_embedding = process_revenue_data(query)
                        D, I = st.session_state.faiss_index.search(query_embedding, 1)
                        
                        # Generate analysis
                        context = st.session_state.documents[I[0][0]]
                        analysis = generate_revenue_analysis(context, query)
                        
                        if analysis:
                            st.markdown("### Analysis Results")
                            st.markdown(analysis)
                    else:
                        st.warning("Please enter your question for analysis.")
                
                # Tips section
                with st.expander("Tips for Better Results", expanded=False):
                    st.markdown("""
                    - Include specific timeframes in your query
                    - Mention relevant sectors or regions
                    - Be specific about metrics you're interested in
                    - Compare multiple factors for comprehensive insights
                    """)
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")