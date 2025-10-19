"""
Anomaly Detection Module
========================

Handles anomaly detection for job market analytics data.
Implements various anomaly detection techniques for different data types.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from collections import Counter
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from scipy import stats
import re

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    A class to detect anomalies in job market data.
    """
    
    def __init__(self):
        """Initialize the AnomalyDetector."""
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        
    def detect_salary_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect salary anomalies using statistical and ML methods.
        
        Args:
            df: DataFrame with salary data
            
        Returns:
            Dict[str, Any]: Anomaly detection results
        """
        if df.empty or 'salary_min' not in df.columns or 'salary_max' not in df.columns:
            return {'error': 'No salary data available'}
        
        logger.info("Detecting salary anomalies...")
        
        # Calculate average salary
        df['avg_salary'] = (df['salary_min'] + df['salary_max']) / 2
        salary_data = df['avg_salary'].dropna()
        
        if len(salary_data) < 10:
            return {'error': 'Insufficient salary data for anomaly detection'}
        
        anomalies = {}
        
        # 1. Statistical Anomalies (Z-score method)
        z_scores = np.abs(stats.zscore(salary_data))
        statistical_anomalies = df[z_scores > 3].copy()
        anomalies['statistical_anomalies'] = {
            'count': len(statistical_anomalies),
            'percentage': len(statistical_anomalies) / len(df) * 100,
            'records': statistical_anomalies[['job_title_clean', 'company_name', 'avg_salary', 'source']].to_dict('records')
        }
        
        # 2. IQR Method
        Q1 = salary_data.quantile(0.25)
        Q3 = salary_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        iqr_anomalies = df[(df['avg_salary'] < lower_bound) | (df['avg_salary'] > upper_bound)].copy()
        anomalies['iqr_anomalies'] = {
            'count': len(iqr_anomalies),
            'percentage': len(iqr_anomalies) / len(df) * 100,
            'bounds': {'lower': lower_bound, 'upper': upper_bound},
            'records': iqr_anomalies[['job_title_clean', 'company_name', 'avg_salary', 'source']].to_dict('records')
        }
        
        # 3. Machine Learning Anomalies (Isolation Forest)
        try:
            # Prepare features for ML
            features = df[['avg_salary', 'experience']].fillna(0)
            features_scaled = self.scaler.fit_transform(features)
            
            # Fit Isolation Forest
            anomaly_labels = self.isolation_forest.fit_predict(features_scaled)
            ml_anomalies = df[anomaly_labels == -1].copy()
            
            anomalies['ml_anomalies'] = {
                'count': len(ml_anomalies),
                'percentage': len(ml_anomalies) / len(df) * 100,
                'records': ml_anomalies[['job_title_clean', 'company_name', 'avg_salary', 'experience', 'source']].to_dict('records')
            }
        except Exception as e:
            logger.warning(f"ML anomaly detection failed: {e}")
            anomalies['ml_anomalies'] = {'error': str(e)}
        
        # 4. Salary vs Job Title Anomalies
        title_salary_anomalies = self._detect_title_salary_anomalies(df)
        anomalies['title_salary_anomalies'] = title_salary_anomalies
        
        # 5. Salary vs Location Anomalies
        location_salary_anomalies = self._detect_location_salary_anomalies(df)
        anomalies['location_salary_anomalies'] = location_salary_anomalies
        
        return anomalies
    
    def _detect_title_salary_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies in salary vs job title relationships."""
        if 'job_title_clean' not in df.columns:
            return {'error': 'No job title data available'}
        
        # Calculate average salary by job title
        title_salaries = df.groupby('job_title_clean')['avg_salary'].agg(['mean', 'std', 'count']).reset_index()
        title_salaries = title_salaries[title_salaries['count'] >= 5]  # At least 5 samples
        
        anomalies = []
        for _, row in title_salaries.iterrows():
            title = row['job_title_clean']
            mean_salary = row['mean']
            std_salary = row['std']
            
            if pd.notna(std_salary) and std_salary > 0:
                # Find jobs with salary > 2 standard deviations from mean
                title_jobs = df[df['job_title_clean'] == title]
                high_salary_jobs = title_jobs[title_jobs['avg_salary'] > mean_salary + 2 * std_salary]
                low_salary_jobs = title_jobs[title_jobs['avg_salary'] < mean_salary - 2 * std_salary]
                
                for _, job in high_salary_jobs.iterrows():
                    anomalies.append({
                        'type': 'high_salary',
                        'job_title': title,
                        'salary': job['avg_salary'],
                        'expected_range': f"{mean_salary - 2*std_salary:.0f} - {mean_salary + 2*std_salary:.0f}",
                        'company': job.get('company_name', ''),
                        'source': job.get('source', '')
                    })
                
                for _, job in low_salary_jobs.iterrows():
                    anomalies.append({
                        'type': 'low_salary',
                        'job_title': title,
                        'salary': job['avg_salary'],
                        'expected_range': f"{mean_salary - 2*std_salary:.0f} - {mean_salary + 2*std_salary:.0f}",
                        'company': job.get('company_name', ''),
                        'source': job.get('source', '')
                    })
        
        return {
            'count': len(anomalies),
            'anomalies': anomalies[:20]  # Limit to top 20
        }
    
    def _detect_location_salary_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies in salary vs location relationships."""
        if 'city' not in df.columns:
            return {'error': 'No location data available'}
        
        # Calculate average salary by city
        city_salaries = df.groupby('city')['avg_salary'].agg(['mean', 'std', 'count']).reset_index()
        city_salaries = city_salaries[city_salaries['count'] >= 5]  # At least 5 samples
        
        anomalies = []
        for _, row in city_salaries.iterrows():
            city = row['city']
            mean_salary = row['mean']
            std_salary = row['std']
            
            if pd.notna(std_salary) and std_salary > 0:
                # Find jobs with salary > 2 standard deviations from city mean
                city_jobs = df[df['city'] == city]
                high_salary_jobs = city_jobs[city_jobs['avg_salary'] > mean_salary + 2 * std_salary]
                low_salary_jobs = city_jobs[city_jobs['avg_salary'] < mean_salary - 2 * std_salary]
                
                for _, job in high_salary_jobs.iterrows():
                    anomalies.append({
                        'type': 'high_salary',
                        'city': city,
                        'salary': job['avg_salary'],
                        'expected_range': f"{mean_salary - 2*std_salary:.0f} - {mean_salary + 2*std_salary:.0f}",
                        'job_title': job.get('job_title_clean', ''),
                        'company': job.get('company_name', ''),
                        'source': job.get('source', '')
                    })
                
                for _, job in low_salary_jobs.iterrows():
                    anomalies.append({
                        'type': 'low_salary',
                        'city': city,
                        'salary': job['avg_salary'],
                        'expected_range': f"{mean_salary - 2*std_salary:.0f} - {mean_salary + 2*std_salary:.0f}",
                        'job_title': job.get('job_title_clean', ''),
                        'company': job.get('company_name', ''),
                        'source': job.get('source', '')
                    })
        
        return {
            'count': len(anomalies),
            'anomalies': anomalies[:20]  # Limit to top 20
        }
    
    def detect_duplicate_jobs(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect duplicate or very similar job postings.
        
        Args:
            df: DataFrame with job data
            
        Returns:
            Dict[str, Any]: Duplicate detection results
        """
        if df.empty:
            return {'error': 'No data available'}
        
        logger.info("Detecting duplicate jobs...")
        
        duplicates = {}
        
        # 1. Exact duplicates (same title, company, location)
        if all(col in df.columns for col in ['job_title_clean', 'company_name', 'city']):
            exact_duplicates = df.groupby(['job_title_clean', 'company_name', 'city']).size()
            exact_duplicates = exact_duplicates[exact_duplicates > 1]
            
            duplicates['exact_duplicates'] = {
                'count': len(exact_duplicates),
                'total_duplicate_records': exact_duplicates.sum(),
                'duplicate_groups': exact_duplicates.to_dict()
            }
        
        # 2. Similar job titles (using string similarity)
        if 'job_title_clean' in df.columns:
            similar_titles = self._find_similar_titles(df)
            duplicates['similar_titles'] = similar_titles
        
        # 3. Same company, similar job descriptions
        if 'job_description' in df.columns and 'company_name' in df.columns:
            similar_descriptions = self._find_similar_descriptions(df)
            duplicates['similar_descriptions'] = similar_descriptions
        
        return duplicates
    
    def _find_similar_titles(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Find similar job titles using string similarity."""
        titles = df['job_title_clean'].dropna().unique()
        similar_groups = []
        
        # Simple similarity based on common words
        for i, title1 in enumerate(titles):
            for j, title2 in enumerate(titles[i+1:], i+1):
                # Calculate Jaccard similarity
                words1 = set(title1.lower().split())
                words2 = set(title2.lower().split())
                
                if len(words1) > 0 and len(words2) > 0:
                    jaccard_sim = len(words1.intersection(words2)) / len(words1.union(words2))
                    
                    if jaccard_sim > 0.7:  # 70% similarity threshold
                        similar_groups.append({
                            'title1': title1,
                            'title2': title2,
                            'similarity': jaccard_sim,
                            'count1': len(df[df['job_title_clean'] == title1]),
                            'count2': len(df[df['job_title_clean'] == title2])
                        })
        
        return {
            'count': len(similar_groups),
            'similar_groups': similar_groups[:20]  # Limit to top 20
        }
    
    def _find_similar_descriptions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Find jobs with similar descriptions from the same company."""
        similar_descriptions = []
        
        # Group by company
        for company, company_df in df.groupby('company_name'):
            if len(company_df) < 2:
                continue
            
            descriptions = company_df['job_description'].dropna()
            if len(descriptions) < 2:
                continue
            
            # Simple similarity based on description length and common words
            for i, desc1 in enumerate(descriptions):
                for j, desc2 in enumerate(descriptions.iloc[i+1:], i+1):
                    if pd.notna(desc1) and pd.notna(desc2):
                        # Calculate similarity based on length ratio and common words
                        len_ratio = min(len(desc1), len(desc2)) / max(len(desc1), len(desc2))
                        
                        words1 = set(desc1.lower().split()[:50])  # First 50 words
                        words2 = set(desc2.lower().split()[:50])
                        
                        if len(words1) > 0 and len(words2) > 0:
                            word_similarity = len(words1.intersection(words2)) / len(words1.union(words2))
                            
                            # Combined similarity score
                            combined_similarity = (len_ratio + word_similarity) / 2
                            
                            if combined_similarity > 0.6:  # 60% similarity threshold
                                similar_descriptions.append({
                                    'company': company,
                                    'similarity': combined_similarity,
                                    'desc1_length': len(desc1),
                                    'desc2_length': len(desc2),
                                    'common_words': len(words1.intersection(words2))
                                })
        
        return {
            'count': len(similar_descriptions),
            'similar_descriptions': similar_descriptions[:20]  # Limit to top 20
        }
    
    def detect_pattern_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect pattern-based anomalies in job postings.
        
        Args:
            df: DataFrame with job data
            
        Returns:
            Dict[str, Any]: Pattern anomaly detection results
        """
        if df.empty:
            return {'error': 'No data available'}
        
        logger.info("Detecting pattern anomalies...")
        
        pattern_anomalies = {}
        
        # 1. Unusual job title patterns
        if 'job_title_clean' in df.columns:
            title_patterns = self._analyze_title_patterns(df)
            pattern_anomalies['title_patterns'] = title_patterns
        
        # 2. Unusual company patterns
        if 'company_name' in df.columns:
            company_patterns = self._analyze_company_patterns(df)
            pattern_anomalies['company_patterns'] = company_patterns
        
        # 3. Unusual skill combinations
        if 'skills' in df.columns:
            skill_patterns = self._analyze_skill_patterns(df)
            pattern_anomalies['skill_patterns'] = skill_patterns
        
        # 4. Unusual experience requirements
        if 'experience' in df.columns:
            experience_patterns = self._analyze_experience_patterns(df)
            pattern_anomalies['experience_patterns'] = experience_patterns
        
        return pattern_anomalies
    
    def _analyze_title_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze unusual job title patterns."""
        titles = df['job_title_clean'].dropna()
        
        # Find titles with unusual characters or patterns
        unusual_titles = []
        for title in titles:
            # Check for unusual patterns
            if re.search(r'[!@#$%^&*()_+=\[\]{}|;:,.<>?]', title):
                unusual_titles.append({
                    'title': title,
                    'pattern': 'unusual_characters',
                    'count': len(df[df['job_title_clean'] == title])
                })
            
            # Check for very long titles
            if len(title) > 100:
                unusual_titles.append({
                    'title': title,
                    'pattern': 'very_long_title',
                    'count': len(df[df['job_title_clean'] == title])
                })
            
            # Check for titles with numbers in unusual positions
            if re.search(r'^\d+', title) or re.search(r'\d{4,}', title):
                unusual_titles.append({
                    'title': title,
                    'pattern': 'unusual_numbers',
                    'count': len(df[df['job_title_clean'] == title])
                })
        
        return {
            'count': len(unusual_titles),
            'unusual_titles': unusual_titles[:20]
        }
    
    def _analyze_company_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze unusual company patterns."""
        companies = df['company_name'].dropna()
        
        # Find companies with unusual patterns
        unusual_companies = []
        for company in companies:
            # Check for companies with very long names
            if len(company) > 50:
                unusual_companies.append({
                    'company': company,
                    'pattern': 'very_long_name',
                    'count': len(df[df['company_name'] == company])
                })
            
            # Check for companies with unusual characters
            if re.search(r'[!@#$%^&*()_+=\[\]{}|;:,.<>?]', company):
                unusual_companies.append({
                    'company': company,
                    'pattern': 'unusual_characters',
                    'count': len(df[df['company_name'] == company])
                })
        
        return {
            'count': len(unusual_companies),
            'unusual_companies': unusual_companies[:20]
        }
    
    def _analyze_skill_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze unusual skill patterns."""
        skills_data = df['skills'].dropna()
        
        unusual_skills = []
        for skills_str in skills_data:
            if isinstance(skills_str, str):
                skills = [skill.strip() for skill in skills_str.split(',')]
                
                # Check for unusual skill combinations
                if len(skills) > 20:  # Too many skills
                    unusual_skills.append({
                        'skills': skills_str[:100] + '...' if len(skills_str) > 100 else skills_str,
                        'pattern': 'too_many_skills',
                        'skill_count': len(skills)
                    })
                
                # Check for skills with unusual characters
                for skill in skills:
                    if re.search(r'[!@#$%^&*()_+=\[\]{}|;:,.<>?]', skill):
                        unusual_skills.append({
                            'skills': skill,
                            'pattern': 'unusual_characters',
                            'skill_count': len(skills)
                        })
        
        return {
            'count': len(unusual_skills),
            'unusual_skills': unusual_skills[:20]
        }
    
    def _analyze_experience_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze unusual experience patterns."""
        experience_data = df['experience'].dropna()
        
        unusual_experience = []
        for exp in experience_data:
            # Check for unrealistic experience values
            if exp > 50:  # More than 50 years experience
                unusual_experience.append({
                    'experience': exp,
                    'pattern': 'unrealistic_high',
                    'count': len(df[df['experience'] == exp])
                })
            
            if exp < 0:  # Negative experience
                unusual_experience.append({
                    'experience': exp,
                    'pattern': 'negative_experience',
                    'count': len(df[df['experience'] == exp])
                })
        
        return {
            'count': len(unusual_experience),
            'unusual_experience': unusual_experience
        }
    
    def generate_anomaly_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive anomaly detection report.
        
        Args:
            df: DataFrame with job data
            
        Returns:
            Dict[str, Any]: Comprehensive anomaly report
        """
        if df.empty:
            return {'error': 'No data available for anomaly detection'}
        
        logger.info("Generating comprehensive anomaly detection report...")
        
        report = {
            'summary': {
                'total_records': len(df),
                'analysis_date': datetime.now().isoformat(),
                'anomaly_types': []
            },
            'salary_anomalies': self.detect_salary_anomalies(df),
            'duplicate_jobs': self.detect_duplicate_jobs(df),
            'pattern_anomalies': self.detect_pattern_anomalies(df)
        }
        
        # Calculate overall anomaly summary
        total_anomalies = 0
        anomaly_types = []
        
        # Count salary anomalies
        if 'salary_anomalies' in report and 'error' not in report['salary_anomalies']:
            for anomaly_type in ['statistical_anomalies', 'iqr_anomalies', 'ml_anomalies']:
                if anomaly_type in report['salary_anomalies']:
                    count = report['salary_anomalies'][anomaly_type].get('count', 0)
                    total_anomalies += count
                    if count > 0:
                        anomaly_types.append(f"{anomaly_type}: {count}")
        
        # Count duplicate jobs
        if 'duplicate_jobs' in report and 'error' not in report['duplicate_jobs']:
            if 'exact_duplicates' in report['duplicate_jobs']:
                count = report['duplicate_jobs']['exact_duplicates'].get('count', 0)
                total_anomalies += count
                if count > 0:
                    anomaly_types.append(f"exact_duplicates: {count}")
        
        # Count pattern anomalies
        if 'pattern_anomalies' in report and 'error' not in report['pattern_anomalies']:
            for pattern_type in ['title_patterns', 'company_patterns', 'skill_patterns']:
                if pattern_type in report['pattern_anomalies']:
                    count = report['pattern_anomalies'][pattern_type].get('count', 0)
                    total_anomalies += count
                    if count > 0:
                        anomaly_types.append(f"{pattern_type}: {count}")
        
        report['summary']['total_anomalies'] = total_anomalies
        report['summary']['anomaly_rate'] = total_anomalies / len(df) * 100 if len(df) > 0 else 0
        report['summary']['anomaly_types'] = anomaly_types
        
        logger.info(f"Anomaly detection report generated: {total_anomalies} anomalies found")
        return report


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Create sample data
    sample_data = pd.DataFrame({
        'job_title_clean': ['Data Scientist', 'Software Engineer', 'Data Scientist', 'Senior Data Scientist'],
        'company_name': ['Google', 'Microsoft', 'Google', 'Amazon'],
        'city': ['San Francisco', 'Seattle', 'San Francisco', 'Seattle'],
        'salary_min': [80000, 70000, 120000, 90000],
        'salary_max': [120000, 100000, 180000, 130000],
        'experience': [3, 2, 5, 4],
        'skills': ['python,machine learning', 'java,spring', 'python,deep learning', 'python,ml,aws'],
        'job_description': ['Analyze data', 'Develop software', 'Advanced data analysis', 'Senior data role'],
        'source': ['glassdoor', 'monster', 'glassdoor', 'naukri']
    })
    
    # Initialize detector
    detector = AnomalyDetector()
    
    # Generate anomaly report
    report = detector.generate_anomaly_report(sample_data)
    
    print("Anomaly Detection Report:")
    print(f"Total anomalies: {report['summary']['total_anomalies']}")
    print(f"Anomaly rate: {report['summary']['anomaly_rate']:.2f}%")
