#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analytics Results Checker
=========================

This script checks and displays the analytics results stored in the database.
"""

import sys
import os
import pandas as pd
import psycopg2
import logging
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalyticsChecker:
    """Checks analytics results from database."""
    
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
    
    def check_processed_jobs(self) -> Dict[str, Any]:
        """Check processed jobs data."""
        try:
            # Basic statistics
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM processed_jobs")
            total_jobs = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT company_name) FROM processed_jobs")
            unique_companies = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT location) FROM processed_jobs")
            unique_locations = cursor.fetchone()[0]
            
            cursor.execute("SELECT source, COUNT(*) FROM processed_jobs GROUP BY source")
            source_distribution = dict(cursor.fetchall())
            
            # Top job titles
            cursor.execute("""
                SELECT job_title, COUNT(*) as count 
                FROM processed_jobs 
                GROUP BY job_title 
                ORDER BY count DESC 
                LIMIT 10
            """)
            top_job_titles = dict(cursor.fetchall())
            
            # Top companies
            cursor.execute("""
                SELECT company_name, COUNT(*) as count 
                FROM processed_jobs 
                WHERE company_name IS NOT NULL AND company_name != ''
                GROUP BY company_name 
                ORDER BY count DESC 
                LIMIT 10
            """)
            top_companies = dict(cursor.fetchall())
            
            cursor.close()
            
            return {
                'total_jobs': total_jobs,
                'unique_companies': unique_companies,
                'unique_locations': unique_locations,
                'source_distribution': source_distribution,
                'top_job_titles': top_job_titles,
                'top_companies': top_companies
            }
            
        except Exception as e:
            logger.error(f"Error checking processed jobs: {e}")
            raise
    
    def check_salary_analysis(self) -> Dict[str, Any]:
        """Check salary analysis results."""
        try:
            cursor = self.conn.cursor()
            
            # Salary analysis records
            cursor.execute("SELECT COUNT(*) FROM salary_analysis")
            salary_records = cursor.fetchone()[0]
            
            # Top paying job titles
            cursor.execute("""
                SELECT job_title, avg_salary, sample_size 
                FROM salary_analysis 
                ORDER BY avg_salary DESC 
                LIMIT 10
            """)
            top_paying_jobs = cursor.fetchall()
            
            cursor.close()
            
            return {
                'salary_records': salary_records,
                'top_paying_jobs': top_paying_jobs
            }
            
        except Exception as e:
            logger.error(f"Error checking salary analysis: {e}")
            raise
    
    def check_skills_analysis(self) -> Dict[str, Any]:
        """Check skills analysis results."""
        try:
            cursor = self.conn.cursor()
            
            # Skills analysis records
            cursor.execute("SELECT COUNT(*) FROM skills_analysis")
            skills_records = cursor.fetchone()[0]
            
            # Top skills
            cursor.execute("""
                SELECT skill_name, frequency 
                FROM skills_analysis 
                ORDER BY frequency DESC 
                LIMIT 15
            """)
            top_skills = cursor.fetchall()
            
            cursor.close()
            
            return {
                'skills_records': skills_records,
                'top_skills': top_skills
            }
            
        except Exception as e:
            logger.error(f"Error checking skills analysis: {e}")
            raise
    
    def check_market_trends(self) -> Dict[str, Any]:
        """Check market trends results."""
        try:
            cursor = self.conn.cursor()
            
            # Market trends records
            cursor.execute("SELECT COUNT(*) FROM market_trends")
            trends_records = cursor.fetchone()[0]
            
            # Market metrics
            cursor.execute("""
                SELECT metric_name, metric_value, category 
                FROM market_trends 
                ORDER BY metric_value DESC
            """)
            market_metrics = cursor.fetchall()
            
            cursor.close()
            
            return {
                'trends_records': trends_records,
                'market_metrics': market_metrics
            }
            
        except Exception as e:
            logger.error(f"Error checking market trends: {e}")
            raise
    
    def get_database_views(self) -> Dict[str, Any]:
        """Check database views."""
        try:
            cursor = self.conn.cursor()
            
            # Job summary view
            cursor.execute("SELECT * FROM v_job_summary LIMIT 10")
            job_summary = cursor.fetchall()
            
            # Location analysis view
            cursor.execute("SELECT * FROM v_location_analysis LIMIT 10")
            location_analysis = cursor.fetchall()
            
            # Industry analysis view
            cursor.execute("SELECT * FROM v_industry_analysis LIMIT 10")
            industry_analysis = cursor.fetchall()
            
            cursor.close()
            
            return {
                'job_summary': job_summary,
                'location_analysis': location_analysis,
                'industry_analysis': industry_analysis
            }
            
        except Exception as e:
            logger.error(f"Error checking database views: {e}")
            raise
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive analytics report."""
        logger.info("Generating comprehensive analytics report...")
        
        report = {
            'processed_jobs': self.check_processed_jobs(),
            'salary_analysis': self.check_salary_analysis(),
            'skills_analysis': self.check_skills_analysis(),
            'market_trends': self.check_market_trends(),
            'database_views': self.get_database_views()
        }
        
        logger.info("Comprehensive analytics report generated")
        return report


def main():
    """Main function to check analytics results."""
    print("Analytics Results Checker")
    print("=" * 50)
    
    try:
        checker = AnalyticsChecker()
        checker.connect()
        
        # Generate comprehensive report
        report = checker.generate_comprehensive_report()
        
        # Display results
        print("\n📊 PROCESSED JOBS ANALYSIS")
        print("-" * 30)
        jobs_data = report['processed_jobs']
        print(f"Total Jobs: {jobs_data['total_jobs']:,}")
        print(f"Unique Companies: {jobs_data['unique_companies']:,}")
        print(f"Unique Locations: {jobs_data['unique_locations']:,}")
        print(f"Source Distribution: {jobs_data['source_distribution']}")
        
        print("\nTop Job Titles:")
        for title, count in list(jobs_data['top_job_titles'].items())[:5]:
            print(f"  • {title}: {count:,} jobs")
        
        print("\nTop Companies:")
        for company, count in list(jobs_data['top_companies'].items())[:5]:
            print(f"  • {company}: {count:,} jobs")
        
        print("\n💰 SALARY ANALYSIS")
        print("-" * 30)
        salary_data = report['salary_analysis']
        print(f"Salary Analysis Records: {salary_data['salary_records']}")
        
        if salary_data['top_paying_jobs']:
            print("\nTop Paying Job Titles:")
            for title, salary, count in salary_data['top_paying_jobs'][:5]:
                print(f"  • {title}: ${salary:,.0f} (based on {count} samples)")
        
        print("\n🔧 SKILLS ANALYSIS")
        print("-" * 30)
        skills_data = report['skills_analysis']
        print(f"Skills Analysis Records: {skills_data['skills_records']}")
        
        if skills_data['top_skills']:
            print("\nTop In-Demand Skills:")
            for skill, frequency in skills_data['top_skills'][:10]:
                print(f"  • {skill}: {frequency:,} mentions")
        
        print("\n📈 MARKET TRENDS")
        print("-" * 30)
        trends_data = report['market_trends']
        print(f"Market Trends Records: {trends_data['trends_records']}")
        
        if trends_data['market_metrics']:
            print("\nMarket Metrics:")
            for metric, value, category in trends_data['market_metrics']:
                print(f"  • {metric}: {value:,} ({category})")
        
        print("\n📋 DATABASE VIEWS")
        print("-" * 30)
        views_data = report['database_views']
        
        if views_data['job_summary']:
            print("\nJob Summary (Top 5):")
            for row in views_data['job_summary'][:5]:
                salary_min = f"${row[2]:,.0f}" if row[2] is not None else "N/A"
                salary_max = f"${row[3]:,.0f}" if row[3] is not None else "N/A"
                print(f"  • {row[0]}: {row[1]:,} jobs, Avg Salary: {salary_min}-{salary_max}")
        
        if views_data['location_analysis']:
            print("\nLocation Analysis (Top 5):")
            for row in views_data['location_analysis'][:5]:
                print(f"  • {row[0]}, {row[1]}: {row[2]:,} jobs")
        
        if views_data['industry_analysis']:
            print("\nIndustry Analysis (Top 5):")
            for row in views_data['industry_analysis'][:5]:
                salary_min = f"${row[2]:,.0f}" if row[2] is not None else "N/A"
                salary_max = f"${row[3]:,.0f}" if row[3] is not None else "N/A"
                print(f"  • {row[0]}: {row[1]:,} jobs, Avg Salary: {salary_min}-{salary_max}")
        
        print("\n✅ Analytics Results Check Completed!")
        print("\n🎯 KEY INSIGHTS:")
        print("-" * 20)
        print(f"• Total jobs processed: {jobs_data['total_jobs']:,}")
        print(f"• Most common job title: {list(jobs_data['top_job_titles'].keys())[0]}")
        print(f"• Most active company: {list(jobs_data['top_companies'].keys())[0]}")
        if skills_data['top_skills']:
            print(f"• Most in-demand skill: {skills_data['top_skills'][0][0]}")
        if salary_data['top_paying_jobs']:
            print(f"• Highest paying job: {salary_data['top_paying_jobs'][0][0]} (${salary_data['top_paying_jobs'][0][1]:,.0f})")
        
    except Exception as e:
        print(f"❌ Analytics check failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'checker' in locals():
            checker.disconnect()


if __name__ == "__main__":
    main()
