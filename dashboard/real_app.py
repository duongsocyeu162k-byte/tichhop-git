"""
Real Data Streamlit Dashboard for Job Market Analytics
======================================================

Interactive dashboard for visualizing real job market trends and insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import requests
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from etl.data_loader import DataLoader
from etl.data_cleaner import DataCleaner
from analytics.trend_analyzer import TrendAnalyzer

# Page configuration
st.set_page_config(
    page_title="Job Market Analytics Dashboard - Real Data",
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
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_real_data():
    """Load and process real data for the dashboard."""
    try:
        # Initialize components
        loader = DataLoader()
        cleaner = DataCleaner()
        analyzer = TrendAnalyzer()
        
        # Load raw data
        with st.spinner("Loading raw data from CSV files..."):
            raw_data = loader.load_all_sources()
        
        # Clean data
        with st.spinner("Cleaning and standardizing data..."):
            cleaned_data = cleaner.clean_all_data(raw_data)
            standardized_data = cleaner.standardize_columns(cleaned_data)
        
        # Combine all data
        all_data = pd.concat(standardized_data.values(), ignore_index=True)
        
        return all_data, standardized_data, analyzer
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), {}, None

def main():
    """Main dashboard application."""
    
    # Header
    st.markdown('<h1 class="main-header">📊 Job Market Analytics Dashboard - Real Data</h1>', 
                unsafe_allow_html=True)
    
    # Load data
    all_data, source_data, analyzer = load_real_data()
    
    if all_data.empty:
        st.error("No data available. Please check your data files.")
        return
    
    # Success message
    st.markdown("""
    <div class="success-box">
        <h4>✅ Data Successfully Loaded!</h4>
        <p>Real job market data has been loaded and processed from multiple sources.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎛️ Dashboard Controls")
    
    # Data source filter
    st.sidebar.subheader("📊 Data Sources")
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
        if 'salary_min' in filtered_data.columns and 'salary_max' in filtered_data.columns:
            filtered_data['avg_salary'] = (filtered_data['salary_min'] + filtered_data['salary_max']) / 2
            avg_salary = filtered_data['avg_salary'].mean()
            st.metric(
                label="Avg Salary",
                value=f"${avg_salary:,.0f}" if not pd.isna(avg_salary) else "N/A"
            )
        else:
            st.metric(
                label="Avg Salary",
                value="N/A"
            )
    
    # Charts section
    st.subheader("📊 Data Visualizations")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Job Distribution", "Geographic Analysis", "Salary Analysis", "Skills Analysis", "Market Insights"
    ])
    
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
        else:
            st.info("Salary data not available for the selected filters.")
    
    with tab4:
        st.subheader("Skills Analysis")
        
        if analyzer:
            # Get skills analysis
            skills_analysis = analyzer.analyze_skills_trends(filtered_data)
            
            if skills_analysis and 'top_skills' in skills_analysis:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Top skills bar chart
                    top_skills = skills_analysis['top_skills']
                    if top_skills:
                        skills_df = pd.DataFrame(list(top_skills.items()), columns=['Skill', 'Count'])
                        fig = px.bar(
                            skills_df.head(15),
                            x='Count',
                            y='Skill',
                            orientation='h',
                            title="Top 15 Skills",
                            labels={'Count': 'Mentions', 'Skill': 'Skill'}
                        )
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Skills summary
                    st.metric("Total Unique Skills", skills_analysis.get('total_unique_skills', 0))
                    st.metric("Total Skill Mentions", sum(skills_analysis.get('top_skills', {}).values()))
            else:
                st.info("No skills data available.")
        else:
            st.info("Skills analyzer not available.")
    
    with tab5:
        st.subheader("Market Insights")
        
        if analyzer:
            # Get market insights
            insights = analyzer.get_market_insights(filtered_data)
            
            if insights:
                for i, insight in enumerate(insights, 1):
                    st.write(f"**{i}.** {insight}")
            else:
                st.info("No insights available.")
        else:
            st.info("Trend analyzer not available.")
    
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
    
    # Data source information
    st.subheader("📊 Data Sources Information")
    
    if source_data:
        for source, df in source_data.items():
            if not df.empty:
                st.write(f"**{source.upper()}**: {len(df):,} records")
    
    # Footer
    st.markdown("---")
    st.markdown("**Job Market Analytics Dashboard** - Powered by real data from multiple sources")

if __name__ == "__main__":
    main()
