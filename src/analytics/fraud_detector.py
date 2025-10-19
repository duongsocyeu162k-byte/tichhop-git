"""
Fraud Detection Module
=====================

Handles fraud detection for job market analytics data.
Detects fake job postings, duplicate listings, and suspicious patterns.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import re
from collections import Counter
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib

logger = logging.getLogger(__name__)


class FraudDetector:
    """
    A class to detect fraudulent job postings and suspicious patterns.
    """
    
    def __init__(self):
        """Initialize the FraudDetector."""
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Fraud detection patterns
        self.suspicious_patterns = {
            'fake_company_indicators': [
                r'company\s+\d+',  # Company 123, Company ABC
                r'startup\s+\d+',  # Startup 123
                r'test\s+company',  # Test Company
                r'sample\s+company',  # Sample Company
                r'fake\s+company',  # Fake Company
                r'demo\s+company'   # Demo Company
            ],
            'suspicious_job_titles': [
                r'work\s+from\s+home\s+job',  # Work from home job
                r'earn\s+\$\d+',  # Earn $1000
                r'make\s+money',  # Make money
                r'no\s+experience\s+needed',  # No experience needed
                r'get\s+rich\s+quick',  # Get rich quick
                r'part\s+time\s+job\s+\$\d+'  # Part time job $500
            ],
            'suspicious_descriptions': [
                r'click\s+here\s+to\s+apply',  # Click here to apply
                r'visit\s+our\s+website',  # Visit our website
                r'call\s+now',  # Call now
                r'limited\s+time\s+offer',  # Limited time offer
                r'act\s+now',  # Act now
                r'guaranteed\s+income',  # Guaranteed income
                r'work\s+from\s+home\s+opportunity',  # Work from home opportunity
                r'no\s+interview\s+required'  # No interview required
            ],
            'suspicious_emails': [
                r'@gmail\.com',  # Gmail addresses (often used for fake companies)
                r'@yahoo\.com',  # Yahoo addresses
                r'@hotmail\.com',  # Hotmail addresses
                r'@outlook\.com',  # Outlook addresses
                r'@\w+\.tk',  # .tk domains (often suspicious)
                r'@\w+\.ml',  # .ml domains
                r'@\w+\.ga'   # .ga domains
            ]
        }
        
        # Known legitimate companies (can be expanded)
        self.legitimate_companies = {
            'google', 'microsoft', 'amazon', 'apple', 'meta', 'facebook', 'netflix',
            'oracle', 'salesforce', 'adobe', 'intel', 'cisco', 'ibm', 'uber',
            'airbnb', 'spotify', 'twitter', 'linkedin', 'paypal', 'stripe',
            'tesla', 'spacex', 'nvidia', 'amd', 'qualcomm', 'broadcom'
        }
    
    def detect_fake_job_postings(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect fake job postings using multiple techniques.
        
        Args:
            df: DataFrame with job data
            
        Returns:
            Dict[str, Any]: Fake job detection results
        """
        if df.empty:
            return {'error': 'No data available'}
        
        logger.info("Detecting fake job postings...")
        
        fake_detection_results = {}
        
        # 1. Pattern-based detection
        pattern_results = self._detect_suspicious_patterns(df)
        fake_detection_results['pattern_detection'] = pattern_results
        
        # 2. Company legitimacy check
        company_results = self._check_company_legitimacy(df)
        fake_detection_results['company_legitimacy'] = company_results
        
        # 3. Duplicate detection
        duplicate_results = self._detect_duplicate_postings(df)
        fake_detection_results['duplicate_detection'] = duplicate_results
        
        # 4. ML-based anomaly detection
        ml_results = self._detect_anomalies_ml(df)
        fake_detection_results['ml_anomaly_detection'] = ml_results
        
        # 5. Salary anomaly detection
        salary_results = self._detect_salary_anomalies(df)
        fake_detection_results['salary_anomalies'] = salary_results
        
        # 6. Contact information analysis
        contact_results = self._analyze_contact_information(df)
        fake_detection_results['contact_analysis'] = contact_results
        
        return fake_detection_results
    
    def _detect_suspicious_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect suspicious patterns in job postings."""
        suspicious_jobs = []
        
        for idx, row in df.iterrows():
            job_suspicion_score = 0
            suspicious_reasons = []
            
            # Check job title patterns
            if 'job_title_clean' in df.columns and pd.notna(row.get('job_title_clean')):
                title = str(row['job_title_clean']).lower()
                for pattern in self.suspicious_patterns['suspicious_job_titles']:
                    if re.search(pattern, title):
                        job_suspicion_score += 2
                        suspicious_reasons.append(f"Suspicious job title pattern: {pattern}")
            
            # Check company name patterns
            if 'company_name' in df.columns and pd.notna(row.get('company_name')):
                company = str(row['company_name']).lower()
                for pattern in self.suspicious_patterns['fake_company_indicators']:
                    if re.search(pattern, company):
                        job_suspicion_score += 3
                        suspicious_reasons.append(f"Fake company indicator: {pattern}")
            
            # Check job description patterns
            if 'job_description' in df.columns and pd.notna(row.get('job_description')):
                description = str(row['job_description']).lower()
                for pattern in self.suspicious_patterns['suspicious_descriptions']:
                    if re.search(pattern, description):
                        job_suspicion_score += 1
                        suspicious_reasons.append(f"Suspicious description pattern: {pattern}")
            
            # Check for excessive capitalization
            if 'job_description' in df.columns and pd.notna(row.get('job_description')):
                description = str(row['job_description'])
                caps_ratio = sum(1 for c in description if c.isupper()) / len(description)
                if caps_ratio > 0.3:  # More than 30% caps
                    job_suspicion_score += 1
                    suspicious_reasons.append("Excessive capitalization")
            
            # Check for excessive exclamation marks
            if 'job_description' in df.columns and pd.notna(row.get('job_description')):
                description = str(row['job_description'])
                exclamation_count = description.count('!')
                if exclamation_count > 5:  # More than 5 exclamation marks
                    job_suspicion_score += 1
                    suspicious_reasons.append("Excessive exclamation marks")
            
            if job_suspicion_score > 0:
                suspicious_jobs.append({
                    'index': idx,
                    'suspicion_score': job_suspicion_score,
                    'reasons': suspicious_reasons,
                    'job_title': row.get('job_title_clean', ''),
                    'company_name': row.get('company_name', ''),
                    'source': row.get('source', '')
                })
        
        # Sort by suspicion score
        suspicious_jobs.sort(key=lambda x: x['suspicion_score'], reverse=True)
        
        return {
            'total_suspicious': len(suspicious_jobs),
            'high_risk_jobs': [job for job in suspicious_jobs if job['suspicion_score'] >= 3],
            'medium_risk_jobs': [job for job in suspicious_jobs if job['suspicion_score'] == 2],
            'low_risk_jobs': [job for job in suspicious_jobs if job['suspicion_score'] == 1],
            'all_suspicious_jobs': suspicious_jobs[:50]  # Limit to top 50
        }
    
    def _check_company_legitimacy(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check company legitimacy."""
        if 'company_name' not in df.columns:
            return {'error': 'No company name data available'}
        
        company_analysis = {}
        
        for company in df['company_name'].dropna().unique():
            company_lower = str(company).lower()
            
            # Check if company is in legitimate companies list
            is_legitimate = company_lower in self.legitimate_companies
            
            # Check for suspicious company name patterns
            is_suspicious = False
            suspicious_reasons = []
            
            for pattern in self.suspicious_patterns['fake_company_indicators']:
                if re.search(pattern, company_lower):
                    is_suspicious = True
                    suspicious_reasons.append(f"Matches pattern: {pattern}")
            
            # Check company name length (too short or too long)
            if len(company) < 3:
                is_suspicious = True
                suspicious_reasons.append("Company name too short")
            elif len(company) > 50:
                is_suspicious = True
                suspicious_reasons.append("Company name too long")
            
            # Check for numbers in company name (often suspicious)
            if re.search(r'\d+', company):
                is_suspicious = True
                suspicious_reasons.append("Contains numbers")
            
            company_analysis[company] = {
                'is_legitimate': is_legitimate,
                'is_suspicious': is_suspicious,
                'suspicious_reasons': suspicious_reasons,
                'job_count': len(df[df['company_name'] == company])
            }
        
        # Summary statistics
        total_companies = len(company_analysis)
        legitimate_companies = sum(1 for info in company_analysis.values() if info['is_legitimate'])
        suspicious_companies = sum(1 for info in company_analysis.values() if info['is_suspicious'])
        
        return {
            'total_companies': total_companies,
            'legitimate_companies': legitimate_companies,
            'suspicious_companies': suspicious_companies,
            'legitimacy_rate': legitimate_companies / total_companies * 100 if total_companies > 0 else 0,
            'suspicion_rate': suspicious_companies / total_companies * 100 if total_companies > 0 else 0,
            'company_details': company_analysis
        }
    
    def _detect_duplicate_postings(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect duplicate job postings."""
        if df.empty:
            return {'error': 'No data available'}
        
        # Create hash for each job posting
        job_hashes = []
        for idx, row in df.iterrows():
            # Create hash based on key fields
            hash_string = f"{row.get('job_title_clean', '')}_{row.get('company_name', '')}_{row.get('city', '')}"
            job_hash = hashlib.md5(hash_string.encode()).hexdigest()
            job_hashes.append((idx, job_hash))
        
        # Find duplicates
        hash_counts = Counter(hash_val for _, hash_val in job_hashes)
        duplicate_hashes = {hash_val: count for hash_val, count in hash_counts.items() if count > 1}
        
        duplicate_groups = []
        for hash_val, count in duplicate_hashes.items():
            duplicate_indices = [idx for idx, h in job_hashes if h == hash_val]
            duplicate_groups.append({
                'hash': hash_val,
                'count': count,
                'indices': duplicate_indices,
                'jobs': df.iloc[duplicate_indices][['job_title_clean', 'company_name', 'city', 'source']].to_dict('records')
            })
        
        # Detect similar job descriptions
        similar_descriptions = self._detect_similar_descriptions(df)
        
        return {
            'exact_duplicates': {
                'count': len(duplicate_groups),
                'total_duplicate_records': sum(group['count'] for group in duplicate_groups),
                'duplicate_groups': duplicate_groups[:20]  # Limit to top 20
            },
            'similar_descriptions': similar_descriptions
        }
    
    def _detect_similar_descriptions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect jobs with similar descriptions."""
        if 'job_description' not in df.columns:
            return {'error': 'No job description data available'}
        
        descriptions = df['job_description'].dropna()
        if len(descriptions) < 2:
            return {'error': 'Insufficient descriptions for similarity analysis'}
        
        # Vectorize descriptions
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(descriptions.astype(str))
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            similar_pairs = []
            for i in range(len(similarity_matrix)):
                for j in range(i + 1, len(similarity_matrix)):
                    similarity = similarity_matrix[i][j]
                    if similarity > 0.8:  # 80% similarity threshold
                        similar_pairs.append({
                            'index1': descriptions.index[i],
                            'index2': descriptions.index[j],
                            'similarity': similarity,
                            'description1': str(descriptions.iloc[i])[:200] + '...',
                            'description2': str(descriptions.iloc[j])[:200] + '...'
                        })
            
            # Sort by similarity
            similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)
            
            return {
                'count': len(similar_pairs),
                'similar_pairs': similar_pairs[:20]  # Limit to top 20
            }
            
        except Exception as e:
            logger.error(f"Error in similarity detection: {e}")
            return {'error': str(e)}
    
    def _detect_anomalies_ml(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies using machine learning."""
        if df.empty:
            return {'error': 'No data available'}
        
        try:
            # Prepare features for anomaly detection
            features = self._prepare_anomaly_features(df)
            
            if features.empty:
                return {'error': 'No features available for anomaly detection'}
            
            # Fit isolation forest
            anomaly_labels = self.isolation_forest.fit_predict(features)
            
            # Get anomaly indices
            anomaly_indices = df.index[anomaly_labels == -1].tolist()
            
            anomaly_jobs = []
            for idx in anomaly_indices:
                row = df.loc[idx]
                anomaly_jobs.append({
                    'index': idx,
                    'job_title': row.get('job_title_clean', ''),
                    'company_name': row.get('company_name', ''),
                    'city': row.get('city', ''),
                    'source': row.get('source', '')
                })
            
            return {
                'total_anomalies': len(anomaly_jobs),
                'anomaly_rate': len(anomaly_jobs) / len(df) * 100,
                'anomaly_jobs': anomaly_jobs
            }
            
        except Exception as e:
            logger.error(f"Error in ML anomaly detection: {e}")
            return {'error': str(e)}
    
    def _prepare_anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for anomaly detection."""
        features_df = pd.DataFrame()
        
        # Numerical features
        if 'experience' in df.columns:
            features_df['experience'] = pd.to_numeric(df['experience'], errors='coerce').fillna(0)
        
        if 'salary_min' in df.columns and 'salary_max' in df.columns:
            features_df['avg_salary'] = (pd.to_numeric(df['salary_min'], errors='coerce') + 
                                       pd.to_numeric(df['salary_max'], errors='coerce')) / 2
            features_df['avg_salary'] = features_df['avg_salary'].fillna(0)
        
        # Text length features
        if 'job_description' in df.columns:
            features_df['description_length'] = df['job_description'].astype(str).str.len()
        
        if 'job_title_clean' in df.columns:
            features_df['title_length'] = df['job_title_clean'].astype(str).str.len()
        
        # Skills count
        if 'skills' in df.columns:
            features_df['skills_count'] = df['skills'].astype(str).apply(
                lambda x: len([s.strip() for s in x.split(',') if s.strip()]) if pd.notna(x) and x != 'nan' else 0
            )
        
        # Fill missing values
        features_df = features_df.fillna(0)
        
        # Scale features
        if not features_df.empty:
            features_df = pd.DataFrame(
                self.scaler.fit_transform(features_df),
                columns=features_df.columns,
                index=features_df.index
            )
        
        return features_df
    
    def _detect_salary_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect salary anomalies that might indicate fraud."""
        if 'salary_min' not in df.columns or 'salary_max' not in df.columns:
            return {'error': 'No salary data available'}
        
        # Calculate average salary
        df['avg_salary'] = (pd.to_numeric(df['salary_min'], errors='coerce') + 
                           pd.to_numeric(df['salary_max'], errors='coerce')) / 2
        
        salary_data = df['avg_salary'].dropna()
        if len(salary_data) == 0:
            return {'error': 'No valid salary data'}
        
        # Detect extremely high salaries (potential fraud)
        q99 = salary_data.quantile(0.99)
        extremely_high = df[df['avg_salary'] > q99 * 2]  # More than 2x 99th percentile
        
        # Detect extremely low salaries (potential fraud)
        q1 = salary_data.quantile(0.01)
        extremely_low = df[df['avg_salary'] < q1 / 2]  # Less than half of 1st percentile
        
        # Detect salary ranges that don't make sense
        df['salary_range'] = pd.to_numeric(df['salary_max'], errors='coerce') - pd.to_numeric(df['salary_min'], errors='coerce')
        invalid_ranges = df[df['salary_range'] < 0]  # Max < Min
        
        return {
            'extremely_high_salaries': {
                'count': len(extremely_high),
                'jobs': extremely_high[['job_title_clean', 'company_name', 'avg_salary', 'source']].to_dict('records')
            },
            'extremely_low_salaries': {
                'count': len(extremely_low),
                'jobs': extremely_low[['job_title_clean', 'company_name', 'avg_salary', 'source']].to_dict('records')
            },
            'invalid_salary_ranges': {
                'count': len(invalid_ranges),
                'jobs': invalid_ranges[['job_title_clean', 'company_name', 'salary_min', 'salary_max', 'source']].to_dict('records')
            }
        }
    
    def _analyze_contact_information(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze contact information for suspicious patterns."""
        contact_analysis = {
            'suspicious_emails': [],
            'missing_contact_info': 0,
            'generic_contact_info': 0
        }
        
        # Check for suspicious email patterns in job descriptions
        if 'job_description' in df.columns:
            for idx, description in df['job_description'].items():
                if pd.notna(description):
                    description_str = str(description)
                    
                    # Find email addresses
                    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                    emails = re.findall(email_pattern, description_str)
                    
                    for email in emails:
                        email_lower = email.lower()
                        is_suspicious = False
                        suspicious_reasons = []
                        
                        # Check against suspicious email patterns
                        for pattern in self.suspicious_patterns['suspicious_emails']:
                            if re.search(pattern, email_lower):
                                is_suspicious = True
                                suspicious_reasons.append(f"Matches suspicious pattern: {pattern}")
                        
                        if is_suspicious:
                            contact_analysis['suspicious_emails'].append({
                                'index': idx,
                                'email': email,
                                'reasons': suspicious_reasons,
                                'job_title': df.loc[idx, 'job_title_clean'] if 'job_title_clean' in df.columns else '',
                                'company_name': df.loc[idx, 'company_name'] if 'company_name' in df.columns else ''
                            })
        
        # Count missing contact information
        if 'company_name' in df.columns:
            contact_analysis['missing_contact_info'] = df['company_name'].isna().sum()
        
        # Count generic contact information
        if 'company_name' in df.columns:
            generic_companies = df['company_name'].str.contains('company|corp|inc|ltd', case=False, na=False)
            contact_analysis['generic_contact_info'] = generic_companies.sum()
        
        return contact_analysis
    
    def generate_fraud_detection_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive fraud detection report.
        
        Args:
            df: DataFrame with job data
            
        Returns:
            Dict[str, Any]: Comprehensive fraud detection report
        """
        if df.empty:
            return {'error': 'No data available for fraud detection'}
        
        logger.info("Generating comprehensive fraud detection report...")
        
        # Detect fake job postings
        fraud_results = self.detect_fake_job_postings(df)
        
        # Calculate overall fraud risk
        total_jobs = len(df)
        high_risk_count = len(fraud_results.get('pattern_detection', {}).get('high_risk_jobs', []))
        medium_risk_count = len(fraud_results.get('pattern_detection', {}).get('medium_risk_jobs', []))
        low_risk_count = len(fraud_results.get('pattern_detection', {}).get('low_risk_jobs', []))
        
        total_risky_jobs = high_risk_count + medium_risk_count + low_risk_count
        fraud_rate = total_risky_jobs / total_jobs * 100 if total_jobs > 0 else 0
        
        # Generate insights
        insights = self._generate_fraud_insights(fraud_results, total_jobs)
        
        report = {
            'summary': {
                'total_jobs_analyzed': total_jobs,
                'high_risk_jobs': high_risk_count,
                'medium_risk_jobs': medium_risk_count,
                'low_risk_jobs': low_risk_count,
                'total_risky_jobs': total_risky_jobs,
                'fraud_rate': fraud_rate,
                'analysis_date': datetime.now().isoformat()
            },
            'fraud_detection_results': fraud_results,
            'insights': insights
        }
        
        logger.info(f"Fraud detection report generated: {fraud_rate:.2f}% fraud rate")
        return report
    
    def _generate_fraud_insights(self, fraud_results: Dict, total_jobs: int) -> List[str]:
        """Generate insights from fraud detection analysis."""
        insights = []
        
        # Overall fraud rate insights
        pattern_detection = fraud_results.get('pattern_detection', {})
        high_risk = len(pattern_detection.get('high_risk_jobs', []))
        medium_risk = len(pattern_detection.get('medium_risk_jobs', []))
        low_risk = len(pattern_detection.get('low_risk_jobs', []))
        
        total_risky = high_risk + medium_risk + low_risk
        fraud_rate = total_risky / total_jobs * 100 if total_jobs > 0 else 0
        
        if fraud_rate > 10:
            insights.append(f"High fraud risk detected: {fraud_rate:.1f}% of job postings are suspicious")
        elif fraud_rate > 5:
            insights.append(f"Moderate fraud risk detected: {fraud_rate:.1f}% of job postings are suspicious")
        else:
            insights.append(f"Low fraud risk: {fraud_rate:.1f}% of job postings are suspicious")
        
        # Company legitimacy insights
        company_legitimacy = fraud_results.get('company_legitimacy', {})
        if 'legitimacy_rate' in company_legitimacy:
            legitimacy_rate = company_legitimacy['legitimacy_rate']
            if legitimacy_rate < 50:
                insights.append(f"Low company legitimacy rate: {legitimacy_rate:.1f}% of companies are legitimate")
            else:
                insights.append(f"Good company legitimacy rate: {legitimacy_rate:.1f}% of companies are legitimate")
        
        # Duplicate insights
        duplicate_detection = fraud_results.get('duplicate_detection', {})
        if 'exact_duplicates' in duplicate_detection:
            duplicate_count = duplicate_detection['exact_duplicates']['count']
            if duplicate_count > 0:
                insights.append(f"Found {duplicate_count} groups of exact duplicate job postings")
        
        # Salary anomaly insights
        salary_anomalies = fraud_results.get('salary_anomalies', {})
        if 'extremely_high_salaries' in salary_anomalies:
            high_salary_count = salary_anomalies['extremely_high_salaries']['count']
            if high_salary_count > 0:
                insights.append(f"Found {high_salary_count} jobs with suspiciously high salaries")
        
        return insights


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Create sample data
    sample_data = pd.DataFrame({
        'job_title_clean': ['Data Scientist', 'Work from Home Job $1000', 'Software Engineer'],
        'company_name': ['Google', 'Company 123', 'Microsoft'],
        'city': ['San Francisco', 'Remote', 'Seattle'],
        'job_description': [
            'Join our team for data analysis',
            'Click here to apply! Earn $1000 per week! No experience needed!',
            'Develop software applications'
        ],
        'salary_min': [80000, 1000, 70000],
        'salary_max': [120000, 1000, 100000],
        'source': ['glassdoor', 'monster', 'naukri']
    })
    
    # Initialize detector
    detector = FraudDetector()
    
    # Generate fraud detection report
    report = detector.generate_fraud_detection_report(sample_data)
    
    print("Fraud Detection Report:")
    print(f"Total Jobs: {report['summary']['total_jobs_analyzed']}")
    print(f"High Risk: {report['summary']['high_risk_jobs']}")
    print(f"Fraud Rate: {report['summary']['fraud_rate']:.2f}%")
