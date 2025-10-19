"""
Trend Analyzer Module
====================

Handles trend analysis and insights generation for job market data.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any
from collections import Counter
import re

logger = logging.getLogger(__name__)


class TrendAnalyzer:
    """
    A class to analyze trends in job market data.
    """
    
    def __init__(self):
        """Initialize the TrendAnalyzer."""
        pass
    
    def extract_skills_from_text(self, text: str) -> List[str]:
        """
        Extract skills from job description text.
        
        Args:
            text: Job description text
            
        Returns:
            List[str]: List of extracted skills
        """
        if pd.isna(text) or text == '':
            return []
        
        # Common technical skills
        skills_keywords = [
            'python', 'java', 'javascript', 'sql', 'r', 'scala', 'go', 'c++', 'c#',
            'machine learning', 'data science', 'statistics', 'analytics',
            'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch',
            'tableau', 'power bi', 'excel', 'spss', 'sas',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes',
            'spark', 'hadoop', 'kafka', 'elasticsearch',
            'mysql', 'postgresql', 'mongodb', 'redis',
            'git', 'github', 'gitlab', 'jenkins',
            'agile', 'scrum', 'jira', 'confluence'
        ]
        
        text_lower = str(text).lower()
        found_skills = []
        
        for skill in skills_keywords:
            if skill in text_lower:
                found_skills.append(skill)
        
        return found_skills
    
    def analyze_job_titles(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze job title trends.
        
        Args:
            df: DataFrame with job data
            
        Returns:
            Dict[str, Any]: Job title analysis results
        """
        if df.empty or 'job_title_clean' not in df.columns:
            return {}
        
        # Get job title counts
        title_counts = df['job_title_clean'].value_counts()
        
        # Top job titles
        top_titles = title_counts.head(20).to_dict()
        
        # Job title categories
        title_categories = self._categorize_job_titles(df['job_title_clean'].tolist())
        
        return {
            'total_unique_titles': len(title_counts),
            'top_titles': top_titles,
            'categories': title_categories
        }
    
    def analyze_salary_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze salary trends.
        
        Args:
            df: DataFrame with job data
            
        Returns:
            Dict[str, Any]: Salary analysis results
        """
        if df.empty:
            return {}
        
        salary_analysis = {}
        
        # Basic salary statistics
        if 'salary_min' in df.columns and 'salary_max' in df.columns:
            # Calculate average salary
            df['avg_salary'] = (df['salary_min'] + df['salary_max']) / 2
            
            salary_analysis = {
                'mean_salary': df['avg_salary'].mean(),
                'median_salary': df['avg_salary'].median(),
                'min_salary': df['avg_salary'].min(),
                'max_salary': df['avg_salary'].max(),
                'std_salary': df['avg_salary'].std(),
                'salary_range_25th': df['avg_salary'].quantile(0.25),
                'salary_range_75th': df['avg_salary'].quantile(0.75)
            }
        
        # Salary by job title
        if 'job_title_clean' in df.columns and 'avg_salary' in df.columns:
            salary_by_title = df.groupby('job_title_clean')['avg_salary'].agg(['mean', 'count']).reset_index()
            salary_by_title = salary_by_title[salary_by_title['count'] >= 5]  # At least 5 samples
            salary_by_title = salary_by_title.sort_values('mean', ascending=False)
            salary_analysis['top_paying_titles'] = salary_by_title.head(10).to_dict('records')
        
        # Salary by location
        if 'city' in df.columns and 'avg_salary' in df.columns:
            salary_by_city = df.groupby('city')['avg_salary'].agg(['mean', 'count']).reset_index()
            salary_by_city = salary_by_city[salary_by_city['count'] >= 5]  # At least 5 samples
            salary_by_city = salary_by_city.sort_values('mean', ascending=False)
            salary_analysis['top_paying_cities'] = salary_by_city.head(10).to_dict('records')
        
        return salary_analysis
    
    def analyze_geographic_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze geographic distribution trends.
        
        Args:
            df: DataFrame with job data
            
        Returns:
            Dict[str, Any]: Geographic analysis results
        """
        if df.empty:
            return {}
        
        geo_analysis = {}
        
        # Country analysis
        if 'country' in df.columns:
            country_counts = df['country'].value_counts()
            geo_analysis['countries'] = {
                'distribution': country_counts.to_dict(),
                'top_countries': country_counts.head(10).to_dict()
            }
        
        # City analysis
        if 'city' in df.columns:
            city_counts = df['city'].value_counts()
            geo_analysis['cities'] = {
                'distribution': city_counts.to_dict(),
                'top_cities': city_counts.head(20).to_dict()
            }
        
        # State analysis (if available)
        if 'state' in df.columns:
            state_counts = df['state'].value_counts()
            geo_analysis['states'] = {
                'distribution': state_counts.to_dict(),
                'top_states': state_counts.head(15).to_dict()
            }
        
        return geo_analysis
    
    def analyze_skills_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze skills trends.
        
        Args:
            df: DataFrame with job data
            
        Returns:
            Dict[str, Any]: Skills analysis results
        """
        if df.empty:
            return {}
        
        skills_analysis = {}
        
        # Extract skills from job descriptions
        all_skills = []
        if 'job_description' in df.columns:
            for desc in df['job_description'].dropna():
                skills = self.extract_skills_from_text(desc)
                all_skills.extend(skills)
        
        # Also use existing skills column if available
        if 'skills' in df.columns:
            for skills_str in df['skills'].dropna():
                if isinstance(skills_str, str):
                    skills = [skill.strip().lower() for skill in skills_str.split(',')]
                    all_skills.extend(skills)
        
        # Count skills
        skills_counter = Counter(all_skills)
        
        skills_analysis = {
            'total_unique_skills': len(skills_counter),
            'top_skills': dict(skills_counter.most_common(20)),
            'skills_frequency': dict(skills_counter)
        }
        
        return skills_analysis
    
    def analyze_industry_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze industry trends.
        
        Args:
            df: DataFrame with job data
            
        Returns:
            Dict[str, Any]: Industry analysis results
        """
        if df.empty or 'industry' not in df.columns:
            return {}
        
        industry_analysis = {}
        
        # Industry distribution
        industry_counts = df['industry'].value_counts()
        industry_analysis = {
            'total_industries': len(industry_counts),
            'industry_distribution': industry_counts.to_dict(),
            'top_industries': industry_counts.head(15).to_dict()
        }
        
        # Industry vs salary analysis
        if 'avg_salary' in df.columns:
            industry_salary = df.groupby('industry')['avg_salary'].agg(['mean', 'count']).reset_index()
            industry_salary = industry_salary[industry_salary['count'] >= 5]  # At least 5 samples
            industry_salary = industry_salary.sort_values('mean', ascending=False)
            industry_analysis['top_paying_industries'] = industry_salary.head(10).to_dict('records')
        
        return industry_analysis
    
    def _categorize_job_titles(self, job_titles: List[str]) -> Dict[str, List[str]]:
        """
        Categorize job titles into groups.
        
        Args:
            job_titles: List of job titles
            
        Returns:
            Dict[str, List[str]]: Categorized job titles
        """
        categories = {
            'Data Science': [],
            'Data Analysis': [],
            'Software Engineering': [],
            'Management': [],
            'Other': []
        }
        
        for title in job_titles:
            title_lower = str(title).lower()
            
            if any(keyword in title_lower for keyword in ['data scientist', 'ml engineer', 'ai engineer']):
                categories['Data Science'].append(title)
            elif any(keyword in title_lower for keyword in ['data analyst', 'business analyst', 'research analyst']):
                categories['Data Analysis'].append(title)
            elif any(keyword in title_lower for keyword in ['software', 'developer', 'engineer', 'programmer']):
                categories['Software Engineering'].append(title)
            elif any(keyword in title_lower for keyword in ['manager', 'director', 'lead', 'head']):
                categories['Management'].append(title)
            else:
                categories['Other'].append(title)
        
        return categories
    
    def generate_trend_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive trend report.
        
        Args:
            df: Combined DataFrame with all job data
            
        Returns:
            Dict[str, Any]: Comprehensive trend report
        """
        if df.empty:
            return {'error': 'No data available for analysis'}
        
        logger.info("Generating comprehensive trend report...")
        
        report = {
            'summary': {
                'total_jobs': len(df),
                'unique_companies': df['company_name'].nunique() if 'company_name' in df.columns else 0,
                'unique_locations': df['city'].nunique() if 'city' in df.columns else 0,
                'data_sources': df['source'].value_counts().to_dict() if 'source' in df.columns else {}
            },
            'job_titles': self.analyze_job_titles(df),
            'salary_trends': self.analyze_salary_trends(df),
            'geographic_trends': self.analyze_geographic_trends(df),
            'skills_trends': self.analyze_skills_trends(df),
            'industry_trends': self.analyze_industry_trends(df)
        }
        
        logger.info("Trend report generated successfully")
        return report
    
    def get_market_insights(self, df: pd.DataFrame) -> List[str]:
        """
        Generate market insights from the data.
        
        Args:
            df: Combined DataFrame with all job data
            
        Returns:
            List[str]: List of market insights
        """
        insights = []
        
        if df.empty:
            return ["No data available for insights generation"]
        
        # Total jobs insight
        total_jobs = len(df)
        insights.append(f"Total job postings analyzed: {total_jobs:,}")
        
        # Source distribution insight
        if 'source' in df.columns:
            source_counts = df['source'].value_counts()
            top_source = source_counts.index[0]
            insights.append(f"Primary data source: {top_source} ({source_counts[top_source]:,} jobs)")
        
        # Top job title insight
        if 'job_title_clean' in df.columns:
            top_title = df['job_title_clean'].value_counts().index[0]
            title_count = df['job_title_clean'].value_counts().iloc[0]
            insights.append(f"Most common job title: {top_title} ({title_count:,} postings)")
        
        # Salary insight
        if 'avg_salary' in df.columns:
            avg_salary = df['avg_salary'].mean()
            insights.append(f"Average salary across all positions: ${avg_salary:,.0f}")
        
        # Geographic insight
        if 'country' in df.columns:
            top_country = df['country'].value_counts().index[0]
            country_count = df['country'].value_counts().iloc[0]
            insights.append(f"Most job postings from: {top_country} ({country_count:,} jobs)")
        
        # Skills insight
        if 'skills' in df.columns:
            all_skills = []
            for skills_str in df['skills'].dropna():
                if isinstance(skills_str, str):
                    skills = [skill.strip().lower() for skill in skills_str.split(',')]
                    all_skills.extend(skills)
            
            if all_skills:
                skills_counter = Counter(all_skills)
                top_skill = skills_counter.most_common(1)[0]
                insights.append(f"Most in-demand skill: {top_skill[0]} (mentioned {top_skill[1]} times)")
        
        return insights


if __name__ == "__main__":
    # Example usage
    from data_loader import DataLoader
    from data_cleaner import DataCleaner
    
    # Load and clean data
    loader = DataLoader()
    raw_data = loader.load_all_sources()
    
    cleaner = DataCleaner()
    cleaned_data = cleaner.clean_all_data(raw_data)
    standardized_data = cleaner.standardize_columns(cleaned_data)
    
    # Combine all data
    all_data = pd.concat(standardized_data.values(), ignore_index=True)
    
    # Analyze trends
    analyzer = TrendAnalyzer()
    report = analyzer.generate_trend_report(all_data)
    
    # Print insights
    insights = analyzer.get_market_insights(all_data)
    for insight in insights:
        print(f"• {insight}")