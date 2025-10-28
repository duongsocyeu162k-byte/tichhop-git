#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database Integration Script
===========================

This script saves ETL processed data to PostgreSQL database
and implements comprehensive analytics functionality.
"""

import sys
import os
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, date
from collections import Counter

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from etl.data_loader import DataLoader
from etl.data_cleaner import DataCleaner
from analytics.trend_analyzer import TrendAnalyzer
from analytics.comprehensive_analyzer import ComprehensiveAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database operations for job analytics."""
    
    def __init__(self, host='localhost', port=5432, database='job_analytics', 
                 user='admin', password='password123'):
        """Initialize database connection."""
        self.connection_params = {
            'host': host,
            'port': port,
            'database': database,
            'user': user,
            'password': password
        }
        self.conn = None
    
    def connect(self):
        """Connect to PostgreSQL database."""
        try:
            self.conn = psycopg2.connect(**self.connection_params)
            logger.info("Connected to PostgreSQL database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def disconnect(self):
        """Disconnect from database."""
        if self.conn:
            self.conn.close()
            logger.info("Disconnected from database")
    
    def save_processed_jobs(self, df: pd.DataFrame):
        """Save processed job data to database."""
        if df.empty:
            logger.warning("No data to save")
            return
        
        try:
            cursor = self.conn.cursor()
            
            # Helper to coerce potentially large numeric to safe int or None
            def _to_int_or_none(value: Any, upper_bound: int = 10_000_000_000) -> Optional[int]:
                try:
                    if value is None or (isinstance(value, float) and np.isnan(value)):
                        return None
                    v = int(float(value))
                    if v <= 0:
                        return None
                    # Drop obviously unrealistic values
                    return v if v <= upper_bound else None
                except Exception:
                    return None

            # Prepare data for insertion
            data_to_insert = []
            for _, row in df.iterrows():
                # Convert skills string to array
                skills_array = []
                if pd.notna(row.get('skills')) and row['skills']:
                    skills_array = [skill.strip() for skill in str(row['skills']).split(',') if skill.strip()]
                
                # Truncate long strings to avoid index issues
                location_clean = str(row.get('location_clean', ''))[:200]  # Limit to 200 chars
                job_description = str(row.get('job_description', ''))[:1000]  # Limit to 1000 chars
                
                # Fallback rule: if only one side provided, set both to that value
                salary_min = row.get('salary_min')
                salary_max = row.get('salary_max')
                if pd.notna(salary_min) and pd.isna(salary_max):
                    salary_max = salary_min
                if pd.notna(salary_max) and pd.isna(salary_min):
                    salary_min = salary_max
                # Sanitize values and cap to prevent DB overflow
                salary_min = _to_int_or_none(salary_min)
                salary_max = _to_int_or_none(salary_max)

                data_tuple = (
                    row.get('source', 'unknown'),
                    str(row.get('job_title_clean', ''))[:100],  # Limit job title
                    str(row.get('company_name', ''))[:100],  # Limit company name
                    location_clean,
                    str(row.get('country', ''))[:50],  # Limit country
                    salary_min,
                    salary_max,
                    'VND',  # Vietnamese datasets -> store as VND
                    str(row.get('industry', ''))[:100],  # Limit industry
                    job_description,
                    skills_array,
                    int(row['experience']) if pd.notna(row.get('experience')) else None,
                    # Extra fields mapped to table columns
                    str(row.get('job_type')) if 'job_type' in df.columns else None,
                    str(row.get('education')) if 'education' in df.columns else None,
                    # Preserve raw salary/experience text if table has columns
                    str(row.get('salary_text')) if 'salary_text' in df.columns else None,
                    str(row.get('experience_text')) if 'experience_text' in df.columns else None,
                    datetime.now()
                )
                data_to_insert.append(data_tuple)
            
            # Insert data using execute_values for better performance
            insert_query = """
                INSERT INTO processed_jobs 
                (source, job_title, company_name, location, country, 
                 salary_min, salary_max, salary_currency, industry, 
                 job_description, skills, experience_years, job_type, 
                 education_level, salary_text, experience_text, created_at)
                VALUES %s
                ON CONFLICT DO NOTHING
            """
            
            execute_values(cursor, insert_query, data_to_insert, page_size=1000)
            self.conn.commit()
            
            logger.info(f"Saved {len(data_to_insert)} processed jobs to database")
            
        except Exception as e:
            logger.error(f"Error saving processed jobs: {e}")
            self.conn.rollback()
            raise
        finally:
            if cursor:
                cursor.close()
    
    def get_processed_jobs(self, limit: int = None) -> pd.DataFrame:
        """Retrieve processed job data from database."""
        try:
            query = "SELECT * FROM processed_jobs"
            if limit:
                query += f" LIMIT {limit}"
            
            df = pd.read_sql_query(query, self.conn)
            logger.info(f"Retrieved {len(df)} processed jobs from database")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving processed jobs: {e}")
            raise
    
    def save_analytics_results(self, analysis_type: str, results: Dict[str, Any]):
        """Save analytics results to appropriate tables."""
        try:
            cursor = self.conn.cursor()
            
            if analysis_type == 'salary_analysis':
                self._save_salary_analysis(cursor, results)
            elif analysis_type == 'skills_analysis':
                self._save_skills_analysis(cursor, results)
            elif analysis_type == 'market_trends':
                self._save_market_trends(cursor, results)
            
            self.conn.commit()
            logger.info(f"Saved {analysis_type} results to database")
            
        except Exception as e:
            logger.error(f"Error saving {analysis_type}: {e}")
            self.conn.rollback()
            raise
        finally:
            if cursor:
                cursor.close()
    
    def _save_salary_analysis(self, cursor, results: Dict[str, Any]):
        """Save salary analysis results."""
        if 'top_paying_titles' in results:
            for item in results['top_paying_titles']:
                cursor.execute("""
                    INSERT INTO salary_analysis 
                    (job_title, avg_salary, min_salary, max_salary, sample_size, analysis_date)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                """, (
                    item['job_title_clean'],
                    item['mean'],
                    item['mean'] * 0.8,  # Estimate min
                    item['mean'] * 1.2,  # Estimate max
                    item['count'],
                    date.today()
                ))
    
    def _save_skills_analysis(self, cursor, results: Dict[str, Any]):
        """Save skills analysis results."""
        if 'top_skills' in results:
            for skill, frequency in results['top_skills'].items():
                cursor.execute("""
                    INSERT INTO skills_analysis 
                    (skill_name, frequency, analysis_date)
                    VALUES (%s, %s, %s)
                    ON CONFLICT DO NOTHING
                """, (skill, frequency, date.today()))
    
    def _save_market_trends(self, cursor, results: Dict[str, Any]):
        """Save market trends results."""
        # Save various trend metrics
        metrics = [
            ('total_jobs', results.get('summary', {}).get('total_jobs', 0), 'overview', 'count'),
            ('unique_companies', results.get('summary', {}).get('unique_companies', 0), 'overview', 'count'),
            ('unique_locations', results.get('summary', {}).get('unique_locations', 0), 'overview', 'count'),
        ]
        
        for metric_name, metric_value, category, subcategory in metrics:
            cursor.execute("""
                INSERT INTO market_trends 
                (metric_name, metric_value, category, subcategory, analysis_date)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (metric_name, metric_value, category, subcategory, date.today()))


class EnhancedTrendAnalyzer(TrendAnalyzer):
    """Enhanced Trend Analyzer with additional functionality."""
    
    def __init__(self):
        super().__init__()
    
    def analyze_job_growth_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze job growth trends over time.
        
        Args:
            df: DataFrame with job data
            
        Returns:
            Dict[str, Any]: Job growth analysis results
        """
        if df.empty:
            return {}
        
        growth_analysis = {}
        
        # Analyze by source (proxy for time if we don't have actual dates)
        if 'source' in df.columns:
            source_counts = df['source'].value_counts()
            growth_analysis['source_distribution'] = source_counts.to_dict()
            
            # Calculate growth rates between sources
            sources = source_counts.index.tolist()
            if len(sources) >= 2:
                growth_rates = {}
                for i in range(1, len(sources)):
                    current_source = sources[i]
                    prev_source = sources[i-1]
                    current_count = source_counts[current_source]
                    prev_count = source_counts[prev_source]
                    
                    growth_rate = ((current_count - prev_count) / prev_count) * 100
                    growth_rates[f"{prev_source}_to_{current_source}"] = growth_rate
                
                growth_analysis['growth_rates'] = growth_rates
        
        # Analyze by job title categories
        if 'job_title_clean' in df.columns:
            title_categories = self._categorize_job_titles(df['job_title_clean'].tolist())
            category_counts = {cat: len(titles) for cat, titles in title_categories.items()}
            growth_analysis['category_distribution'] = category_counts
        
        return growth_analysis
    
    def analyze_industry_trends_detailed(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detailed industry trend analysis.
        
        Args:
            df: DataFrame with job data
            
        Returns:
            Dict[str, Any]: Detailed industry analysis
        """
        if df.empty or 'industry' not in df.columns:
            return {}
        
        industry_analysis = {}
        
        # Basic industry distribution
        industry_counts = df['industry'].value_counts()
        industry_analysis['industry_distribution'] = industry_counts.to_dict()
        industry_analysis['top_industries'] = industry_counts.head(20).to_dict()
        
        # Industry growth analysis
        if 'source' in df.columns:
            industry_by_source = df.groupby(['source', 'industry']).size().unstack(fill_value=0)
            industry_analysis['industry_by_source'] = industry_by_source.to_dict()
        
        # Industry vs salary analysis
        if 'salary_min' in df.columns and 'salary_max' in df.columns:
            df['avg_salary'] = (df['salary_min'] + df['salary_max']) / 2
            industry_salary = df.groupby('industry')['avg_salary'].agg(['mean', 'median', 'count']).reset_index()
            industry_salary = industry_salary[industry_salary['count'] >= 5]
            industry_salary = industry_salary.sort_values('mean', ascending=False)
            industry_analysis['industry_salary_analysis'] = industry_salary.to_dict('records')
        
        # Industry vs skills analysis
        if 'skills' in df.columns:
            industry_skills = {}
            for industry in df['industry'].unique():
                if pd.notna(industry):
                    industry_df = df[df['industry'] == industry]
                    all_skills = []
                    for skills_str in industry_df['skills'].dropna():
                        if isinstance(skills_str, str):
                            skills = [skill.strip().lower() for skill in skills_str.split(',')]
                            all_skills.extend(skills)
                    
                    if all_skills:
                        skills_counter = Counter(all_skills)
                        industry_skills[industry] = dict(skills_counter.most_common(10))
            
            industry_analysis['industry_skills'] = industry_skills
        
        return industry_analysis
    
    def analyze_geographic_distribution_detailed(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detailed geographic distribution analysis.
        
        Args:
            df: DataFrame with job data
            
        Returns:
            Dict[str, Any]: Detailed geographic analysis
        """
        if df.empty:
            return {}
        
        geo_analysis = {}
        
        # Country analysis
        if 'country' in df.columns:
            country_counts = df['country'].value_counts()
            geo_analysis['country_distribution'] = country_counts.to_dict()
            geo_analysis['top_countries'] = country_counts.head(10).to_dict()
        
        # City analysis with salary data
        if 'city' in df.columns and 'salary_min' in df.columns and 'salary_max' in df.columns:
            df['avg_salary'] = (df['salary_min'] + df['salary_max']) / 2
            city_analysis = df.groupby('city').agg({
                'avg_salary': ['mean', 'count'],
                'job_title_clean': 'count'
            }).reset_index()
            
            city_analysis.columns = ['city', 'avg_salary', 'salary_count', 'job_count']
            city_analysis = city_analysis[city_analysis['salary_count'] >= 5]
            city_analysis = city_analysis.sort_values('avg_salary', ascending=False)
            geo_analysis['city_salary_analysis'] = city_analysis.head(20).to_dict('records')
        
        # Geographic clustering
        if 'city' in df.columns and 'country' in df.columns:
            geo_clusters = df.groupby(['country', 'city']).size().reset_index(name='job_count')
            geo_clusters = geo_clusters.sort_values('job_count', ascending=False)
            geo_analysis['geographic_clusters'] = geo_clusters.head(30).to_dict('records')
        
        return geo_analysis
    
    def analyze_skills_trends_detailed(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detailed skills trend analysis.
        
        Args:
            df: DataFrame with job data
            
        Returns:
            Dict[str, Any]: Detailed skills analysis
        """
        if df.empty:
            return {}
        
        skills_analysis = {}
        
        # Extract all skills
        all_skills = []
        if 'skills' in df.columns:
            for skills_str in df['skills'].dropna():
                if isinstance(skills_str, str):
                    skills = [skill.strip().lower() for skill in skills_str.split(',')]
                    all_skills.extend(skills)
        
        if not all_skills:
            return {'error': 'No skills data available'}
        
        # Skills frequency analysis
        skills_counter = Counter(all_skills)
        skills_analysis['skills_frequency'] = dict(skills_counter)
        skills_analysis['top_skills'] = dict(skills_counter.most_common(30))
        
        # Skills by industry
        if 'industry' in df.columns:
            industry_skills = {}
            for industry in df['industry'].unique():
                if pd.notna(industry):
                    industry_df = df[df['industry'] == industry]
                    industry_skills_list = []
                    for skills_str in industry_df['skills'].dropna():
                        if isinstance(skills_str, str):
                            skills = [skill.strip().lower() for skill in skills_str.split(',')]
                            industry_skills_list.extend(skills)
                    
                    if industry_skills_list:
                        industry_skills_counter = Counter(industry_skills_list)
                        industry_skills[industry] = dict(industry_skills_counter.most_common(10))
            
            skills_analysis['skills_by_industry'] = industry_skills
        
        # Skills by salary
        if 'salary_min' in df.columns and 'salary_max' in df.columns:
            df['avg_salary'] = (df['salary_min'] + df['salary_max']) / 2
            
            skill_salary_analysis = {}
            for skill in skills_counter.most_common(20):
                skill_name = skill[0]
                skill_jobs = df[df['skills'].str.contains(skill_name, case=False, na=False)]
                if len(skill_jobs) >= 5:
                    avg_salary = skill_jobs['avg_salary'].mean()
                    skill_salary_analysis[skill_name] = {
                        'avg_salary': avg_salary,
                        'job_count': len(skill_jobs)
                    }
            
            skills_analysis['skills_salary_analysis'] = skill_salary_analysis
        
        return skills_analysis
    
    def analyze_salary_trends_detailed(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detailed salary trend analysis.
        
        Args:
            df: DataFrame with job data
            
        Returns:
            Dict[str, Any]: Detailed salary analysis
        """
        if df.empty:
            return {}
        
        salary_analysis = {}
        
        # Calculate average salary
        if 'salary_min' in df.columns and 'salary_max' in df.columns:
            df['avg_salary'] = (df['salary_min'] + df['salary_max']) / 2
            
            # Overall salary statistics
            salary_stats = {
                'mean_salary': df['avg_salary'].mean(),
                'median_salary': df['avg_salary'].median(),
                'min_salary': df['avg_salary'].min(),
                'max_salary': df['avg_salary'].max(),
                'std_salary': df['avg_salary'].std(),
                'salary_percentiles': {
                    '25th': df['avg_salary'].quantile(0.25),
                    '50th': df['avg_salary'].quantile(0.50),
                    '75th': df['avg_salary'].quantile(0.75),
                    '90th': df['avg_salary'].quantile(0.90),
                    '95th': df['avg_salary'].quantile(0.95)
                }
            }
            salary_analysis['overall_statistics'] = salary_stats
            
            # Salary by job title
            if 'job_title_clean' in df.columns:
                title_salary = df.groupby('job_title_clean')['avg_salary'].agg(['mean', 'median', 'count']).reset_index()
                title_salary = title_salary[title_salary['count'] >= 5]
                title_salary = title_salary.sort_values('mean', ascending=False)
                salary_analysis['salary_by_title'] = title_salary.head(20).to_dict('records')
            
            # Salary by location
            if 'city' in df.columns:
                city_salary = df.groupby('city')['avg_salary'].agg(['mean', 'median', 'count']).reset_index()
                city_salary = city_salary[city_salary['count'] >= 5]
                city_salary = city_salary.sort_values('mean', ascending=False)
                salary_analysis['salary_by_location'] = city_salary.head(20).to_dict('records')
            
            # Salary by industry
            if 'industry' in df.columns:
                industry_salary = df.groupby('industry')['avg_salary'].agg(['mean', 'median', 'count']).reset_index()
                industry_salary = industry_salary[industry_salary['count'] >= 5]
                industry_salary = industry_salary.sort_values('mean', ascending=False)
                salary_analysis['salary_by_industry'] = industry_salary.head(20).to_dict('records')
            
            # Salary by experience
            if 'experience' in df.columns:
                exp_salary = df.groupby('experience')['avg_salary'].agg(['mean', 'count']).reset_index()
                exp_salary = exp_salary[exp_salary['count'] >= 5]
                exp_salary = exp_salary.sort_values('experience')
                salary_analysis['salary_by_experience'] = exp_salary.to_dict('records')
        
        return salary_analysis
    
    def generate_comprehensive_analytics_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive analytics report with all trend analyses.
        
        Args:
            df: Combined DataFrame with all job data
            
        Returns:
            Dict[str, Any]: Comprehensive analytics report
        """
        if df.empty:
            return {'error': 'No data available for analysis'}
        
        logger.info("Generating comprehensive analytics report...")
        
        report = {
            'summary': {
                'total_jobs': len(df),
                'unique_companies': df['company_name'].nunique() if 'company_name' in df.columns else 0,
                'unique_locations': df['city'].nunique() if 'city' in df.columns else 0,
                'data_sources': df['source'].value_counts().to_dict() if 'source' in df.columns else {},
                'analysis_date': datetime.now().isoformat()
            },
            'job_growth_trends': self.analyze_job_growth_trends(df),
            'industry_trends': self.analyze_industry_trends_detailed(df),
            'geographic_distribution': self.analyze_geographic_distribution_detailed(df),
            'skills_trends': self.analyze_skills_trends_detailed(df),
            'salary_trends': self.analyze_salary_trends_detailed(df)
        }
        
        logger.info("Comprehensive analytics report generated successfully")
        return report


def main():
    """Main function to run the complete pipeline."""
    print("Database Integration and Analytics Pipeline")
    print("=" * 60)
    
    try:
        # Initialize components
        db_manager = DatabaseManager()
        db_manager.connect()
        
        # Load and process data
        print("1. Loading and processing data...")
        loader = DataLoader()
        raw_data = loader.load_all_sources()
        
        # Initialize DataCleaner with MongoDB integration
        mongodb_connection = "mongodb://admin:password123@localhost:27017/job_analytics"
        cleaner = DataCleaner(mongodb_connection_string=mongodb_connection)
        
        # Use comprehensive cleaning and storage method
        print("   Processing data with MongoDB integration...")
        processing_results = cleaner.clean_and_store_data(raw_data, store_in_mongodb=True)
        
        # Extract standardized data for PostgreSQL
        standardized_data = processing_results['data_ready_for_postgresql']
        
        # Print MongoDB storage results
        mongodb_results = processing_results['mongodb_storage']
        if 'error' not in mongodb_results and 'skipped' not in mongodb_results:
            print("   ✅ Data stored in MongoDB successfully")
            if 'raw_data_stored' in mongodb_results:
                print(f"   - Raw data: {sum(mongodb_results['raw_data_stored'].values())} records")
            if 'processed_data_stored' in mongodb_results:
                print(f"   - Processed data: {sum(mongodb_results['processed_data_stored'].values())} records")
            if 'job_descriptions_stored' in mongodb_results:
                print(f"   - Job descriptions: {mongodb_results['job_descriptions_stored']} records")
        else:
            print(f"   ⚠️  MongoDB storage issue: {mongodb_results.get('error', mongodb_results.get('skipped', 'Unknown error'))}")
        
        # Combine all data
        all_data = []
        for source, df in standardized_data.items():
            if not df.empty:
                df_copy = df.copy()
                df_copy['source'] = source
                all_data.append(df_copy)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"   Combined data: {len(combined_df):,} records")
        else:
            print("   No data to process")
            return
        
        # Save to database
        print("2. Saving processed data to database...")
        db_manager.save_processed_jobs(combined_df)
        
        # Generate comprehensive analytics using all modules
        print("3. Generating comprehensive analytics...")
        comprehensive_analyzer = ComprehensiveAnalyzer()
        analytics_report = comprehensive_analyzer.generate_comprehensive_report(combined_df)
        
        # Also generate legacy trend analysis for compatibility
        legacy_analyzer = EnhancedTrendAnalyzer()
        legacy_report = legacy_analyzer.generate_comprehensive_analytics_report(combined_df)
        
        # Save analytics results
        print("4. Saving analytics results to database...")
        
        # Save legacy analytics for compatibility
        if 'salary_trends' in legacy_report:
            db_manager.save_analytics_results('salary_analysis', legacy_report['salary_trends'])
        if 'skills_trends' in legacy_report:
            db_manager.save_analytics_results('skills_analysis', legacy_report['skills_trends'])
        db_manager.save_analytics_results('market_trends', legacy_report)
        
        # Save comprehensive analytics metadata
        analytics_summary = comprehensive_analyzer.get_analytics_summary(analytics_report)
        db_manager.save_analytics_results('comprehensive_analytics', analytics_summary)
        
        # Display comprehensive analytics summary
        print("\nComprehensive Analytics Summary:")
        print("=" * 50)
        comprehensive_summary = analytics_summary
        print(f"Total Jobs Analyzed: {comprehensive_summary['total_jobs']:,}")
        print(f"Analysis Date: {comprehensive_summary['analysis_date']}")
        
        # Display module status
        print("\nAnalytics Modules Status:")
        print("-" * 30)
        for module, status in comprehensive_summary['modules_status'].items():
            status_icon = "✅" if status == 'success' else "❌"
            print(f"{status_icon} {module.replace('_', ' ').title()}: {status}")
        
        # Display key metrics
        print("\nKey Metrics:")
        print("-" * 20)
        key_metrics = comprehensive_summary['key_metrics']
        if 'anomaly_rate' in key_metrics:
            print(f"Anomaly Rate: {key_metrics['anomaly_rate']:.1f}%")
        if 'fraud_rate' in key_metrics:
            print(f"Fraud Rate: {key_metrics['fraud_rate']:.1f}%")
        if 'positive_sentiment' in key_metrics:
            print(f"Positive Sentiment: {key_metrics['positive_sentiment']:.1f}%")
        if 'salary_model_performance' in key_metrics:
            print(f"Salary Model Performance: {key_metrics['salary_model_performance']:.3f}")
        if 'market_maturity' in key_metrics:
            print(f"Market Maturity: {key_metrics['market_maturity']}")
        if 'growth_potential' in key_metrics:
            print(f"Growth Potential: {key_metrics['growth_potential']}")
        
        # Display alerts
        if comprehensive_summary['alerts']:
            print("\n⚠️  Alerts:")
            print("-" * 20)
            for alert in comprehensive_summary['alerts']:
                print(f"• {alert}")
        
        # Display overall insights
        if 'overall_insights' in analytics_report:
            print("\nOverall Insights:")
            print("-" * 20)
            for insight in analytics_report['overall_insights']:
                print(f"• {insight}")
        
        # Display legacy summary for compatibility
        print("\nLegacy Analytics Summary:")
        print("=" * 30)
        legacy_summary = legacy_report['summary']
        print(f"Total Jobs Analyzed: {legacy_summary['total_jobs']:,}")
        print(f"Unique Companies: {legacy_summary['unique_companies']:,}")
        print(f"Unique Locations: {legacy_summary['unique_locations']:,}")
        print(f"Data Sources: {legacy_summary['data_sources']}")
        
        # Display MongoDB capabilities
        print("\nMongoDB Integration:")
        print("-" * 20)
        mongodb_capabilities = cleaner.get_mongodb_search_capabilities()
        if 'error' not in mongodb_capabilities:
            print(f"✅ MongoDB collections: {len(mongodb_capabilities['collections_available'])}")
            print(f"✅ Total documents: {mongodb_capabilities['total_documents']:,}")
            print(f"✅ Full-text search: {'Enabled' if mongodb_capabilities['full_text_search_enabled'] else 'Disabled'}")
            print(f"✅ Skills search: {'Enabled' if mongodb_capabilities['skills_search_enabled'] else 'Disabled'}")
            print(f"✅ Company search: {'Enabled' if mongodb_capabilities['company_search_enabled'] else 'Disabled'}")
        else:
            print(f"⚠️  MongoDB capabilities: {mongodb_capabilities['error']}")
        
        print("\n✅ Pipeline completed successfully!")
        print("📊 Data is now available in both PostgreSQL and MongoDB!")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'db_manager' in locals():
            db_manager.disconnect()


if __name__ == "__main__":
    main()
