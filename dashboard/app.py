"""
Streamlit Dashboard for Job Market Analytics
============================================

Interactive dashboard for visualizing job market trends and insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from etl.data_loader import DataLoader
from etl.data_cleaner import DataCleaner
from analytics.trend_analyzer import TrendAnalyzer

# Page configuration
st.set_page_config(
    page_title="Job Market Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

def load_data():
    """Load and process data for the dashboard."""
    try:
        # Initialize components
        loader = DataLoader()
        cleaner = DataCleaner()
        analyzer = TrendAnalyzer()
        
        # Load raw data
        raw_data = loader.load_all_sources()
        
        # Clean data
        cleaned_data = cleaner.clean_all_data(raw_data)
        
        # Combine all data
        all_data = pd.concat(cleaned_data.values(), ignore_index=True)
        
        return all_data, cleaned_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), {}

def main():
    """Main dashboard application."""
    
    # Header
    st.markdown('<h1 class="main-header">📊 Job Market Analytics Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎛️ Dashboard Controls")
    
    # Load data
    with st.spinner("Loading data..."):
        all_data, cleaned_data = load_data()
    
    if all_data.empty:
        st.error("No data available. Please check your data files.")
        return
    
    # Sidebar filters
    st.sidebar.subheader("📊 Data Filters")
    
    # Source filter
    sources = all_data['source'].unique() if 'source' in all_data.columns else []
    selected_sources = st.sidebar.multiselect(
        "Select Data Sources",
        sources,
        default=sources
    )
    
    # Filter data based on selection
    if selected_sources:
        filtered_data = all_data[all_data['source'].isin(selected_sources)]
    else:
        filtered_data = all_data
    
    # Job title filter
    if 'job_title_clean' in filtered_data.columns:
        job_titles = filtered_data['job_title_clean'].value_counts().head(20).index
        selected_titles = st.sidebar.multiselect(
            "Select Job Titles",
            job_titles,
            default=job_titles[:5]
        )
        
        if selected_titles:
            filtered_data = filtered_data[filtered_data['job_title_clean'].isin(selected_titles)]
    
    # Location filter
    if 'country' in filtered_data.columns:
        countries = filtered_data['country'].value_counts().head(10).index
        selected_countries = st.sidebar.multiselect(
            "Select Countries",
            countries,
            default=countries[:3]
        )
        
        if selected_countries:
            filtered_data = filtered_data[filtered_data['country'].isin(selected_countries)]
    
    # Main content
    st.subheader("📈 Overview Metrics")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Jobs",
            value=f"{len(filtered_data):,}",
            delta=f"{len(filtered_data) - len(all_data):,}" if len(filtered_data) != len(all_data) else None
        )
    
    with col2:
        unique_companies = filtered_data['company_name'].nunique() if 'company_name' in filtered_data.columns else 0
        st.metric(
            label="Unique Companies",
            value=f"{unique_companies:,}"
        )
    
    with col3:
        unique_locations = filtered_data['city'].nunique() if 'city' in filtered_data.columns else 0
        st.metric(
            label="Unique Locations",
            value=f"{unique_locations:,}"
        )
    
    with col4:
        avg_salary = filtered_data['salary_min'].mean() if 'salary_min' in filtered_data.columns else 0
        st.metric(
            label="Avg Min Salary",
            value=f"${avg_salary:,.0f}" if avg_salary > 0 else "N/A"
        )
    
    # Charts section
    st.subheader("📊 Data Visualizations")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Job Distribution", "Geographic Analysis", "Salary Analysis", "Trends"])
    
    with tab1:
        st.subheader("Job Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Job titles distribution
            if 'job_title_clean' in filtered_data.columns:
                job_counts = filtered_data['job_title_clean'].value_counts().head(10)
                fig = px.bar(
                    x=job_counts.values,
                    y=job_counts.index,
                    orientation='h',
                    title="Top 10 Job Titles",
                    labels={'x': 'Count', 'y': 'Job Title'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Industry distribution
            if 'industry' in filtered_data.columns:
                industry_counts = filtered_data['industry'].value_counts().head(10)
                fig = px.pie(
                    values=industry_counts.values,
                    names=industry_counts.index,
                    title="Top 10 Industries"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Geographic Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Country distribution
            if 'country' in filtered_data.columns:
                country_counts = filtered_data['country'].value_counts().head(10)
                fig = px.bar(
                    x=country_counts.index,
                    y=country_counts.values,
                    title="Jobs by Country",
                    labels={'x': 'Country', 'y': 'Count'}
                )
                fig.update_xaxes(tickangle=45)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # City distribution
            if 'city' in filtered_data.columns:
                city_counts = filtered_data['city'].value_counts().head(15)
                fig = px.bar(
                    x=city_counts.values,
                    y=city_counts.index,
                    orientation='h',
                    title="Top 15 Cities",
                    labels={'x': 'Count', 'y': 'City'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Salary Analysis")
        
        # Salary distribution
        if 'salary_min' in filtered_data.columns and 'salary_max' in filtered_data.columns:
            # Calculate average salary
            filtered_data['avg_salary'] = (filtered_data['salary_min'] + filtered_data['salary_max']) / 2
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Salary histogram
                fig = px.histogram(
                    filtered_data,
                    x='avg_salary',
                    nbins=30,
                    title="Salary Distribution",
                    labels={'avg_salary': 'Average Salary', 'count': 'Frequency'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Salary by job title
                if 'job_title_clean' in filtered_data.columns:
                    salary_by_title = filtered_data.groupby('job_title_clean')['avg_salary'].mean().sort_values(ascending=False).head(10)
                    fig = px.bar(
                        x=salary_by_title.values,
                        y=salary_by_title.index,
                        orientation='h',
                        title="Average Salary by Job Title",
                        labels={'x': 'Average Salary', 'y': 'Job Title'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Market Trends")
        
        # Source comparison
        if 'source' in filtered_data.columns:
            source_counts = filtered_data['source'].value_counts()
            fig = px.pie(
                values=source_counts.values,
                names=source_counts.index,
                title="Data Distribution by Source"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.subheader("📋 Raw Data")
    
    # Show sample data
    st.dataframe(
        filtered_data.head(100),
        use_container_width=True,
        height=400
    )
    
    # Download button
    csv = filtered_data.to_csv(index=False)
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_job_data.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()
