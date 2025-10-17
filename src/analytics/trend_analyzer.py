"""
Trend Analyzer Module
====================

Analyzes trends in job market data including job growth, 
industry trends, and geographic patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from collections import Counter

logger = logging.getLogger(__name__)


class TrendAnalyzer:
    """
    A class to analyze trends in job market data.
    """
    
    def __init__(self):
        """Initialize the TrendAnalyzer."""
        self.trend_data = {}
    
    def analyze_job_growth(self, df: pd.DataFrame, 
                          time_column: str = 'created_at',
                          job_column: str = 'job_title_clean') -> Dict:
        """
        Analyze job growth trends over time.
        
        Args:
            df: DataFrame with job data
            time_column: Column containing time information
            job_column: Column containing job titles
            
        Returns:
            Dict: Job growth analysis results
        """
        logger.info("Analyzing job growth trends...")
        
        # Convert time column to datetime
        if time_column in df.columns:
            df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
        
        # Group by time period and count jobs
        if time_column in df.columns:
            df['month'] = df[time_column].dt.to_period('M')
            monthly_counts = df.groupby('month').size()
            
            # Calculate growth rate
            growth_rate = monthly_counts.pct_change().mean()
            
            # Identify peak months
            peak_month = monthly_counts.idxmax()
            peak_count = monthly_counts.max()
            
            # Trend direction
            if len(monthly_counts) >= 2:
                recent_trend = "increasing" if monthly_counts.iloc[-1] > monthly_counts.iloc[-2] else "decreasing"
            else:
                recent_trend = "stable"
            
            return {
                'monthly_counts': monthly_counts.to_dict(),
                'growth_rate': growth_rate,
                'peak_month': str(peak_month),
                'peak_count': peak_count,
                'recent_trend': recent_trend,
                'total_jobs': len(df)
            }
        else:
            logger.warning(f"Time column '{time_column}' not found in data")
            return {'error': f"Time column '{time_column}' not found"}
    
    def analyze_industry_trends(self, df: pd.DataFrame, 
                               industry_column: str = 'industry') -> Dict:
        """
        Analyze trends by industry.
        
        Args:
            df: DataFrame with job data
            industry_column: Column containing industry information
            
        Returns:
            Dict: Industry trend analysis results
        """
        logger.info("Analyzing industry trends...")
        
        if industry_column not in df.columns:
            logger.warning(f"Industry column '{industry_column}' not found")
            return {'error': f"Industry column '{industry_column}' not found"}
        
        # Count jobs by industry
        industry_counts = df[industry_column].value_counts()
        
        # Calculate percentages
        industry_percentages = (industry_counts / len(df) * 100).round(2)
        
        # Top growing industries (if we have time data)
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
            df['month'] = df['created_at'].dt.to_period('M')
            
            # Compare first half vs second half
            mid_point = len(df['month'].unique()) // 2
            first_half = df[df['month'] <= df['month'].unique()[mid_point]]
            second_half = df[df['month'] > df['month'].unique()[mid_point]]
            
            first_half_counts = first_half[industry_column].value_counts()
            second_half_counts = second_half[industry_column].value_counts()
            
            # Calculate growth
            growth_data = {}
            for industry in industry_counts.index:
                first_count = first_half_counts.get(industry, 0)
                second_count = second_half_counts.get(industry, 0)
                if first_count > 0:
                    growth = ((second_count - first_count) / first_count) * 100
                    growth_data[industry] = growth
                else:
                    growth_data[industry] = 0
        else:
            growth_data = {}
        
        return {
            'industry_counts': industry_counts.to_dict(),
            'industry_percentages': industry_percentages.to_dict(),
            'growth_by_industry': growth_data,
            'top_industries': industry_counts.head(10).to_dict()
        }
    
    def analyze_geographic_trends(self, df: pd.DataFrame,
                                 city_column: str = 'city',
                                 country_column: str = 'country') -> Dict:
        """
        Analyze geographic distribution of jobs.
        
        Args:
            df: DataFrame with job data
            city_column: Column containing city information
            country_column: Column containing country information
            
        Returns:
            Dict: Geographic trend analysis results
        """
        logger.info("Analyzing geographic trends...")
        
        results = {}
        
        # Country analysis
        if country_column in df.columns:
            country_counts = df[country_column].value_counts()
            country_percentages = (country_counts / len(df) * 100).round(2)
            
            results['country_counts'] = country_counts.to_dict()
            results['country_percentages'] = country_percentages.to_dict()
            results['top_countries'] = country_counts.head(10).to_dict()
        
        # City analysis
        if city_column in df.columns:
            city_counts = df[city_column].value_counts()
            city_percentages = (city_counts / len(df) * 100).round(2)
            
            results['city_counts'] = city_counts.to_dict()
            results['city_percentages'] = city_percentages.to_dict()
            results['top_cities'] = city_counts.head(20).to_dict()
        
        # Geographic distribution by source
        if 'source' in df.columns:
            geo_by_source = df.groupby(['source', country_column]).size().unstack(fill_value=0)
            results['geo_by_source'] = geo_by_source.to_dict()
        
        return results
    
    def analyze_skills_trends(self, df: pd.DataFrame,
                              skills_column: str = 'skills') -> Dict:
        """
        Analyze trending skills in job market.
        
        Args:
            df: DataFrame with job data
            skills_column: Column containing skills information
            
        Returns:
            Dict: Skills trend analysis results
        """
        logger.info("Analyzing skills trends...")
        
        if skills_column not in df.columns:
            logger.warning(f"Skills column '{skills_column}' not found")
            return {'error': f"Skills column '{skills_column}' not found"}
        
        # Extract all skills
        all_skills = []
        for skills_str in df[skills_column].dropna():
            if isinstance(skills_str, str):
                # Split by common delimiters
                skills = [skill.strip() for skill in skills_str.split(',')]
                all_skills.extend(skills)
        
        # Count skills
        skills_counter = Counter(all_skills)
        skills_counts = pd.Series(dict(skills_counter))
        
        # Calculate percentages
        skills_percentages = (skills_counts / len(df) * 100).round(2)
        
        # Top skills
        top_skills = skills_counts.head(20)
        
        return {
            'skills_counts': skills_counts.to_dict(),
            'skills_percentages': skills_percentages.to_dict(),
            'top_skills': top_skills.to_dict(),
            'total_unique_skills': len(skills_counts)
        }
    
    def analyze_salary_trends(self, df: pd.DataFrame,
                              salary_min_col: str = 'salary_min',
                              salary_max_col: str = 'salary_max') -> Dict:
        """
        Analyze salary trends over time and by location.
        
        Args:
            df: DataFrame with job data
            salary_min_col: Column containing minimum salary
            salary_max_col: Column containing maximum salary
            
        Returns:
            Dict: Salary trend analysis results
        """
        logger.info("Analyzing salary trends...")
        
        results = {}
        
        # Calculate average salary
        if salary_min_col in df.columns and salary_max_col in df.columns:
            df['avg_salary'] = (df[salary_min_col] + df[salary_max_col]) / 2
            
            # Overall salary statistics
            results['overall_stats'] = {
                'mean_salary': df['avg_salary'].mean(),
                'median_salary': df['avg_salary'].median(),
                'std_salary': df['avg_salary'].std(),
                'min_salary': df['avg_salary'].min(),
                'max_salary': df['avg_salary'].max()
            }
            
            # Salary by location
            if 'city' in df.columns:
                salary_by_city = df.groupby('city')['avg_salary'].agg(['mean', 'count']).sort_values('mean', ascending=False)
                results['salary_by_city'] = salary_by_city.head(20).to_dict()
            
            # Salary by industry
            if 'industry' in df.columns:
                salary_by_industry = df.groupby('industry')['avg_salary'].agg(['mean', 'count']).sort_values('mean', ascending=False)
                results['salary_by_industry'] = salary_by_industry.head(20).to_dict()
            
            # Salary trends over time
            if 'created_at' in df.columns:
                df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
                df['month'] = df['created_at'].dt.to_period('M')
                monthly_salary = df.groupby('month')['avg_salary'].mean()
                results['monthly_salary_trend'] = monthly_salary.to_dict()
        
        return results
    
    def generate_trend_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive trend analysis report.
        
        Args:
            df: DataFrame with job data
            
        Returns:
            Dict: Complete trend analysis report
        """
        logger.info("Generating comprehensive trend report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_records': len(df),
            'job_growth': self.analyze_job_growth(df),
            'industry_trends': self.analyze_industry_trends(df),
            'geographic_trends': self.analyze_geographic_trends(df),
            'skills_trends': self.analyze_skills_trends(df),
            'salary_trends': self.analyze_salary_trends(df)
        }
        
        return report
    
    def plot_trends(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """
        Create visualization plots for trends.
        
        Args:
            df: DataFrame with job data
            save_path: Path to save plots
        """
        logger.info("Creating trend visualizations...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Job Market Trends Analysis', fontsize=16, fontweight='bold')
        
        # 1. Industry distribution
        if 'industry' in df.columns:
            industry_counts = df['industry'].value_counts().head(10)
            axes[0, 0].pie(industry_counts.values, labels=industry_counts.index, autopct='%1.1f%%')
            axes[0, 0].set_title('Top 10 Industries')
        
        # 2. Geographic distribution
        if 'country' in df.columns:
            country_counts = df['country'].value_counts().head(10)
            axes[0, 1].bar(range(len(country_counts)), country_counts.values)
            axes[0, 1].set_xticks(range(len(country_counts)))
            axes[0, 1].set_xticklabels(country_counts.index, rotation=45)
            axes[0, 1].set_title('Jobs by Country')
        
        # 3. Salary distribution
        if 'salary_min' in df.columns and 'salary_max' in df.columns:
            df['avg_salary'] = (df['salary_min'] + df['salary_max']) / 2
            axes[1, 0].hist(df['avg_salary'].dropna(), bins=30, alpha=0.7)
            axes[1, 0].set_title('Salary Distribution')
            axes[1, 0].set_xlabel('Average Salary')
            axes[1, 0].set_ylabel('Frequency')
        
        # 4. Job titles distribution
        if 'job_title_clean' in df.columns:
            job_counts = df['job_title_clean'].value_counts().head(10)
            axes[1, 1].barh(range(len(job_counts)), job_counts.values)
            axes[1, 1].set_yticks(range(len(job_counts)))
            axes[1, 1].set_yticklabels(job_counts.index)
            axes[1, 1].set_title('Top 10 Job Titles')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Trend plots saved to {save_path}")
        
        plt.show()


if __name__ == "__main__":
    # Example usage
    from ..etl.data_loader import DataLoader
    from ..etl.data_cleaner import DataCleaner
    
    # Load and clean data
    loader = DataLoader()
    raw_data = loader.load_all_sources()
    
    cleaner = DataCleaner()
    cleaned_data = cleaner.clean_all_data(raw_data)
    
    # Combine all data
    all_data = pd.concat(cleaned_data.values(), ignore_index=True)
    
    # Analyze trends
    analyzer = TrendAnalyzer()
    report = analyzer.generate_trend_report(all_data)
    
    print("Trend Analysis Report:")
    print(f"Total records: {report['total_records']}")
    print(f"Job growth rate: {report['job_growth'].get('growth_rate', 'N/A')}")
    print(f"Top industries: {list(report['industry_trends'].get('top_industries', {}).keys())[:5]}")
