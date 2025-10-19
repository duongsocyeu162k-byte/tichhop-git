"""
Data Cleaner Module
==================

Handles data cleaning, transformation, and standardization
for job market analytics data.
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import unicodedata
from .schema_matcher import SchemaMatcher, DataMatcher
from .mongodb_storage import MongoDBStorage

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    A class to handle data cleaning and transformation.
    """
    
    def __init__(self, mongodb_connection_string: Optional[str] = None):
        """
        Initialize the DataCleaner.
        
        Args:
            mongodb_connection_string: MongoDB connection string (optional)
        """
        self.salary_patterns = {
            'range': r'\$?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*[-–—]\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)',
            'single': r'\$?(\d+(?:,\d{3})*(?:\.\d{2})?)',
            'k_suffix': r'(\d+(?:,\d{3})*)\s*[Kk]',
            'hourly': r'\$?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*/\s*hr',
            'yearly': r'\$?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*/\s*yr'
        }
        
        # Experience extraction patterns
        self.experience_patterns = [
            r'(\d+)\+?\s*to\s*(\d+)\s*years?',
            r'(\d+)\+?\s*-\s*(\d+)\s*years?',
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'(\d+)\+?\s*years?\s*exp',
            r'(\d+)\+?\s*yrs?\s*exp',
            r'(\d+)\+?\s*years?',
            r'(\d+)\+?\s*yr',
            r'(\d+)\+?\s*y\b'
        ]
        
        # Skills keywords for extraction
        self.skills_keywords = [
            # Programming Languages
            'python', 'java', 'javascript', 'typescript', 'sql', 'r', 'scala', 'go', 'c++', 'c#', 'php', 'ruby',
            'swift', 'kotlin', 'rust', 'dart', 'perl', 'matlab', 'sas', 'stata',
            
            # Data Science & ML
            'machine learning', 'ml', 'deep learning', 'data science', 'data analysis', 'statistics', 'analytics',
            'pandas', 'numpy', 'scipy', 'scikit-learn', 'sklearn', 'tensorflow', 'pytorch', 'keras',
            'xgboost', 'lightgbm', 'catboost', 'spark', 'hadoop', 'kafka', 'elasticsearch',
            
            # Visualization & BI
            'tableau', 'power bi', 'excel', 'spss', 'sas', 'matplotlib', 'seaborn', 'plotly', 'd3.js',
            'looker', 'qlik', 'microstrategy', 'cognos',
            
            # Cloud & DevOps
            'aws', 'amazon web services', 'azure', 'gcp', 'google cloud', 'docker', 'kubernetes', 'jenkins',
            'git', 'github', 'gitlab', 'bitbucket', 'ci/cd', 'terraform', 'ansible',
            
            # Databases
            'mysql', 'postgresql', 'postgres', 'mongodb', 'redis', 'cassandra', 'elasticsearch',
            'oracle', 'sql server', 'sqlite', 'dynamodb', 'bigquery', 'redshift',
            
            # Web Technologies
            'html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask',
            'spring', 'laravel', 'rails', 'fastapi', 'streamlit', 'dash',
            
            # Business & Soft Skills
            'agile', 'scrum', 'jira', 'confluence', 'project management', 'leadership',
            'communication', 'teamwork', 'problem solving', 'critical thinking',
            
            # Domain Specific
            'finance', 'banking', 'healthcare', 'e-commerce', 'retail', 'manufacturing',
            'telecommunications', 'automotive', 'aerospace', 'energy', 'pharmaceuticals'
        ]
        
        # Initialize schema and data matchers
        self.schema_matcher = SchemaMatcher()
        self.data_matcher = DataMatcher()
        
        # Initialize MongoDB storage (optional)
        self.mongodb_storage = None
        if mongodb_connection_string:
            try:
                self.mongodb_storage = MongoDBStorage(mongodb_connection_string)
                logger.info("MongoDB storage initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize MongoDB storage: {e}")
                self.mongodb_storage = None
        
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text data.
        
        Args:
            text: Input text
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to string and strip whitespace
        text = str(text).strip()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\-.,()&/]', '', text)
        
        # Normalize unicode
        text = unicodedata.normalize('NFKD', text)
        
        return text
    
    def extract_salary_range(self, salary_text: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Extract salary range from text.
        
        Args:
            salary_text: Salary text (e.g., "$50K-$70K", "50000-70000")
            
        Returns:
            Tuple[Optional[float], Optional[float]]: (min_salary, max_salary)
        """
        if pd.isna(salary_text) or salary_text == '':
            return None, None
        
        salary_text = str(salary_text).strip()
        
        # Handle K suffix (e.g., "50K-70K")
        if 'K' in salary_text.upper():
            salary_text = re.sub(r'(\d+(?:,\d{3})*)\s*[Kk]', r'\g<1>000', salary_text)
        
        # Extract range pattern
        range_match = re.search(self.salary_patterns['range'], salary_text)
        if range_match:
            min_sal = float(range_match.group(1).replace(',', ''))
            max_sal = float(range_match.group(2).replace(',', ''))
            return min_sal, max_sal
        
        # Extract single value
        single_match = re.search(self.salary_patterns['single'], salary_text)
        if single_match:
            salary = float(single_match.group(1).replace(',', ''))
            return salary, salary
        
        return None, None
    
    def extract_experience(self, text: str) -> Optional[int]:
        """
        Extract years of experience from text.
        
        Args:
            text: Input text (job description, experience field, etc.)
            
        Returns:
            Optional[int]: Years of experience or None if not found
        """
        if pd.isna(text) or text == '':
            return None
        
        text = str(text).lower()
        
        # Try each pattern
        for pattern in self.experience_patterns:
            match = re.search(pattern, text)
            if match:
                # Handle range patterns (e.g., "3-5 years")
                if len(match.groups()) == 2:
                    try:
                        min_exp = int(match.group(1))
                        max_exp = int(match.group(2))
                        return (min_exp + max_exp) // 2  # Return average
                    except ValueError:
                        continue
                else:
                    try:
                        return int(match.group(1))
                    except ValueError:
                        continue
        
        return None
    
    def extract_skills(self, text: str) -> List[str]:
        """
        Extract skills from text.
        
        Args:
            text: Input text (job description, skills field, etc.)
            
        Returns:
            List[str]: List of found skills
        """
        if pd.isna(text) or text == '':
            return []
        
        text_lower = str(text).lower()
        found_skills = []
        
        for skill in self.skills_keywords:
            if skill in text_lower:
                found_skills.append(skill)
        
        return list(set(found_skills))  # Remove duplicates
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality and return validation results.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dict[str, Any]: Validation results
        """
        validation_results = {
            'total_rows': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'validation_errors': [],
            'warnings': []
        }
        
        # Validate salary range logic
        if 'salary_min' in df.columns and 'salary_max' in df.columns:
            # Convert to numeric, coercing errors to NaN
            salary_min_numeric = pd.to_numeric(df['salary_min'], errors='coerce')
            salary_max_numeric = pd.to_numeric(df['salary_max'], errors='coerce')
            
            # Check for invalid salary ranges (min > max)
            invalid_salary = df[(salary_min_numeric.notna()) & (salary_max_numeric.notna()) & 
                              (salary_min_numeric > salary_max_numeric)]
            if len(invalid_salary) > 0:
                validation_results['validation_errors'].append(
                    f"Found {len(invalid_salary)} records with salary_min > salary_max"
                )
        
        # Validate experience range
        if 'experience' in df.columns:
            numeric_exp = pd.to_numeric(df['experience'], errors='coerce')
            invalid_exp = numeric_exp[(numeric_exp < 0) | (numeric_exp > 50)]
            if len(invalid_exp) > 0:
                validation_results['warnings'].append(
                    f"Found {len(invalid_exp)} records with unrealistic experience values"
                )
        
        # Validate required fields completeness
        required_fields = ['job_title_clean', 'company_name', 'source']
        for field in required_fields:
            if field in df.columns:
                missing_count = df[field].isnull().sum()
                if missing_count > 0:
                    validation_results['warnings'].append(
                        f"Field '{field}' has {missing_count} missing values"
                    )
        
        # Validate data types
        if 'salary_min' in df.columns:
            non_numeric_salary = pd.to_numeric(df['salary_min'], errors='coerce').isnull().sum()
            if non_numeric_salary > 0:
                validation_results['validation_errors'].append(
                    f"Found {non_numeric_salary} non-numeric values in salary_min"
                )
        
        return validation_results
    
    def get_data_quality_score(self, df: pd.DataFrame) -> float:
        """
        Calculate data quality score (0-100).
        
        Args:
            df: DataFrame to score
            
        Returns:
            float: Quality score from 0 to 100
        """
        if df.empty:
            return 0.0
        
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        
        # Base score from completeness
        completeness_score = ((total_cells - missing_cells) / total_cells) * 100
        
        # Deduct points for validation errors
        validation_results = self.validate_data(df)
        error_penalty = len(validation_results['validation_errors']) * 5
        warning_penalty = len(validation_results['warnings']) * 2
        
        final_score = max(0, completeness_score - error_penalty - warning_penalty)
        return round(final_score, 2)
    
    def clean_glassdoor_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean Glassdoor data.
        
        Args:
            df: Raw Glassdoor DataFrame
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        if df.empty:
            return df
        
        logger.info("Cleaning Glassdoor data...")
        cleaned_df = df.copy()
        
        # Add source identifier
        cleaned_df['source'] = 'glassdoor'
        
        # Clean job titles
        if 'Job Title' in cleaned_df.columns:
            cleaned_df['job_title_clean'] = cleaned_df['Job Title'].apply(self.clean_text)
        
        # Clean company names
        if 'Company Name' in cleaned_df.columns:
            cleaned_df['company_name'] = cleaned_df['Company Name'].apply(self.clean_text)
        
        # Clean locations
        if 'Location' in cleaned_df.columns:
            cleaned_df['location_clean'] = cleaned_df['Location'].apply(self.clean_text)
            # Extract city and state
            cleaned_df['city'] = cleaned_df['Location'].apply(self._extract_city)
            cleaned_df['state'] = cleaned_df['Location'].apply(self._extract_state)
            cleaned_df['country'] = 'United States'  # Glassdoor is US-focused
        
        # Clean salary estimates
        if 'Salary Estimate' in cleaned_df.columns:
            salary_data = cleaned_df['Salary Estimate'].apply(self.extract_salary_range)
            cleaned_df['salary_min'] = [x[0] for x in salary_data]
            cleaned_df['salary_max'] = [x[1] for x in salary_data]
        
        # Clean industry
        if 'Industry' in cleaned_df.columns:
            cleaned_df['industry'] = cleaned_df['Industry'].apply(self.clean_text)
        
        # Clean job descriptions
        if 'Job Description' in cleaned_df.columns:
            cleaned_df['job_description'] = cleaned_df['Job Description'].apply(self.clean_text)
        
        # Clean ratings
        if 'Rating' in cleaned_df.columns:
            cleaned_df['rating'] = pd.to_numeric(cleaned_df['Rating'], errors='coerce')
        
        # Clean company size
        if 'Size' in cleaned_df.columns:
            cleaned_df['company_size'] = cleaned_df['Size'].apply(self.clean_text)
        
        # Extract skills from job description
        if 'Job Description' in cleaned_df.columns:
            cleaned_df['skills'] = cleaned_df['Job Description'].apply(
                lambda x: ','.join(self.extract_skills(x)) if pd.notna(x) else ''
            )
        else:
            cleaned_df['skills'] = ''
        
        # Extract experience from job description
        if 'Job Description' in cleaned_df.columns:
            cleaned_df['experience'] = cleaned_df['Job Description'].apply(self.extract_experience)
        else:
            cleaned_df['experience'] = None
        
        logger.info(f"Cleaned Glassdoor data: {len(cleaned_df)} rows")
        return cleaned_df
    
    def clean_monster_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean Monster.com data.
        
        Args:
            df: Raw Monster DataFrame
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        if df.empty:
            return df
        
        logger.info("Cleaning Monster data...")
        cleaned_df = df.copy()
        
        # Add source identifier
        cleaned_df['source'] = 'monster'
        
        # Clean job titles
        if 'job_title' in cleaned_df.columns:
            cleaned_df['job_title_clean'] = cleaned_df['job_title'].apply(self.clean_text)
        
        # Clean company names
        if 'organization' in cleaned_df.columns:
            cleaned_df['company_name'] = cleaned_df['organization'].apply(self.clean_text)
        
        # Clean locations
        if 'location' in cleaned_df.columns:
            cleaned_df['location_clean'] = cleaned_df['location'].apply(self.clean_text)
            cleaned_df['city'] = cleaned_df['location'].apply(self._extract_city)
            cleaned_df['state'] = cleaned_df['location'].apply(self._extract_state)
            cleaned_df['country'] = cleaned_df['country'].fillna('Unknown')
        
        # Clean salary
        if 'salary' in cleaned_df.columns:
            salary_data = cleaned_df['salary'].apply(self.extract_salary_range)
            cleaned_df['salary_min'] = [x[0] for x in salary_data]
            cleaned_df['salary_max'] = [x[1] for x in salary_data]
        else:
            cleaned_df['salary_min'] = None
            cleaned_df['salary_max'] = None
        
        # Clean industry/sector
        if 'sector' in cleaned_df.columns:
            cleaned_df['industry'] = cleaned_df['sector'].apply(self.clean_text)
        
        # Clean job descriptions
        if 'job_description' in cleaned_df.columns:
            cleaned_df['job_description'] = cleaned_df['job_description'].apply(self.clean_text)
        
        # Add missing columns
        cleaned_df['rating'] = None
        cleaned_df['company_size'] = None
        
        # Extract skills from job description
        if 'job_description' in cleaned_df.columns:
            cleaned_df['skills'] = cleaned_df['job_description'].apply(
                lambda x: ','.join(self.extract_skills(x)) if pd.notna(x) else ''
            )
        else:
            cleaned_df['skills'] = ''
        
        # Extract experience from job description
        if 'job_description' in cleaned_df.columns:
            cleaned_df['experience'] = cleaned_df['job_description'].apply(self.extract_experience)
        else:
            cleaned_df['experience'] = None
        
        logger.info(f"Cleaned Monster data: {len(cleaned_df)} rows")
        return cleaned_df
    
    def clean_naukri_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean Naukri.com data.
        
        Args:
            df: Raw Naukri DataFrame
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        if df.empty:
            return df
        
        logger.info("Cleaning Naukri data...")
        cleaned_df = df.copy()
        
        # Add source identifier
        cleaned_df['source'] = 'naukri'
        
        # Clean job titles
        if 'jobtitle' in cleaned_df.columns:
            cleaned_df['job_title_clean'] = cleaned_df['jobtitle'].apply(self.clean_text)
        
        # Clean company names
        if 'company' in cleaned_df.columns:
            cleaned_df['company_name'] = cleaned_df['company'].apply(self.clean_text)
        
        # Clean locations
        if 'joblocation_address' in cleaned_df.columns:
            cleaned_df['location_clean'] = cleaned_df['joblocation_address'].apply(self.clean_text)
            cleaned_df['city'] = cleaned_df['joblocation_address'].apply(self._extract_city)
            cleaned_df['state'] = cleaned_df['joblocation_address'].apply(self._extract_state)
            cleaned_df['country'] = 'India'  # Naukri is India-focused
        
        # Clean salary
        if 'payrate' in cleaned_df.columns:
            salary_data = cleaned_df['payrate'].apply(self.extract_salary_range)
            cleaned_df['salary_min'] = [x[0] for x in salary_data]
            cleaned_df['salary_max'] = [x[1] for x in salary_data]
        else:
            cleaned_df['salary_min'] = None
            cleaned_df['salary_max'] = None
        
        # Clean industry
        if 'industry' in cleaned_df.columns:
            cleaned_df['industry'] = cleaned_df['industry'].apply(self.clean_text)
        
        # Clean job descriptions
        if 'jobdescription' in cleaned_df.columns:
            cleaned_df['job_description'] = cleaned_df['jobdescription'].apply(self.clean_text)
        
        # Clean and extract skills
        if 'skills' in cleaned_df.columns:
            # First clean existing skills
            cleaned_df['skills'] = cleaned_df['skills'].apply(self.clean_text)
            # Then extract additional skills from job description
            if 'jobdescription' in cleaned_df.columns:
                additional_skills = cleaned_df['jobdescription'].apply(
                    lambda x: ','.join(self.extract_skills(x)) if pd.notna(x) else ''
                )
                # Combine existing and extracted skills
                cleaned_df['skills'] = cleaned_df['skills'] + ',' + additional_skills
                cleaned_df['skills'] = cleaned_df['skills'].apply(
                    lambda x: ','.join(list(set([s.strip() for s in x.split(',') if s.strip()])))
                )
        else:
            # Extract skills from job description only
            if 'jobdescription' in cleaned_df.columns:
                cleaned_df['skills'] = cleaned_df['jobdescription'].apply(
                    lambda x: ','.join(self.extract_skills(x)) if pd.notna(x) else ''
                )
            else:
                cleaned_df['skills'] = ''
        
        # Clean and extract experience
        if 'experience' in cleaned_df.columns:
            # First try to extract from existing experience field
            cleaned_df['experience'] = cleaned_df['experience'].apply(self.extract_experience)
            # Then try to extract from job description if not found
            if 'jobdescription' in cleaned_df.columns:
                job_exp = cleaned_df['jobdescription'].apply(self.extract_experience)
                cleaned_df['experience'] = cleaned_df['experience'].fillna(job_exp)
        else:
            # Extract experience from job description only
            if 'jobdescription' in cleaned_df.columns:
                cleaned_df['experience'] = cleaned_df['jobdescription'].apply(self.extract_experience)
            else:
                cleaned_df['experience'] = None
        
        # Add missing columns
        cleaned_df['rating'] = None
        cleaned_df['company_size'] = None
        
        logger.info(f"Cleaned Naukri data: {len(cleaned_df)} rows")
        return cleaned_df
    
    def _extract_city(self, location: str) -> str:
        """Extract city from location string."""
        if pd.isna(location):
            return ''
        
        location = str(location).strip()
        # Split by comma and take first part
        parts = location.split(',')
        if parts:
            return parts[0].strip()
        return location
    
    def _extract_state(self, location: str) -> str:
        """Extract state from location string."""
        if pd.isna(location):
            return ''
        
        location = str(location).strip()
        # Split by comma and take second part
        parts = location.split(',')
        if len(parts) > 1:
            return parts[1].strip()
        return ''
    
    def clean_all_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Clean all data sources.
        
        Args:
            data: Dictionary of raw DataFrames
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of cleaned DataFrames
        """
        cleaned_data = {}
        
        for source, df in data.items():
            if source == 'glassdoor':
                cleaned_data[source] = self.clean_glassdoor_data(df)
            elif source == 'monster':
                cleaned_data[source] = self.clean_monster_data(df)
            elif source == 'naukri':
                cleaned_data[source] = self.clean_naukri_data(df)
            else:
                logger.warning(f"Unknown data source: {source}")
                cleaned_data[source] = df
        
        return cleaned_data
    
    def standardize_columns(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Standardize column names across all data sources.
        
        Args:
            data: Dictionary of DataFrames
            
        Returns:
            Dict[str, pd.DataFrame]: DataFrames with standardized columns
        """
        standardized_data = {}
        
        # Define standard columns
        standard_columns = [
            'source', 'job_title_clean', 'company_name', 'location_clean',
            'city', 'state', 'country', 'salary_min', 'salary_max',
            'industry', 'job_description', 'rating', 'company_size',
            'skills', 'experience'
        ]
        
        for source, df in data.items():
            if df.empty:
                standardized_data[source] = df
                continue
            
            # Create standardized DataFrame
            std_df = pd.DataFrame()
            
            for col in standard_columns:
                if col in df.columns:
                    std_df[col] = df[col]
                else:
                    std_df[col] = None
            
            standardized_data[source] = std_df
            logger.info(f"Standardized {source} data: {len(std_df)} rows")
        
        return standardized_data
    
    def get_cleaning_summary(self, original_data: Dict[str, pd.DataFrame], 
                           cleaned_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Get comprehensive summary of cleaning process.
        
        Args:
            original_data: Original data before cleaning
            cleaned_data: Data after cleaning
            
        Returns:
            Dict[str, Dict]: Comprehensive cleaning summary
        """
        summary = {}
        
        for source in original_data.keys():
            orig_df = original_data[source]
            clean_df = cleaned_data[source]
            
            # Basic cleaning stats
            basic_stats = {
                'original_rows': len(orig_df),
                'cleaned_rows': len(clean_df),
                'rows_removed': len(orig_df) - len(clean_df),
                'columns_standardized': len(clean_df.columns),
                'missing_values': clean_df.isnull().sum().to_dict()
            }
            
            # Data quality assessment
            quality_score = self.get_data_quality_score(clean_df)
            validation_results = self.validate_data(clean_df)
            
            # Skills and experience extraction stats
            skills_stats = {}
            if 'skills' in clean_df.columns:
                non_empty_skills = clean_df['skills'].apply(lambda x: len(str(x).strip()) > 0).sum()
                skills_stats = {
                    'records_with_skills': int(non_empty_skills),
                    'skills_extraction_rate': round(non_empty_skills / len(clean_df) * 100, 2) if len(clean_df) > 0 else 0
                }
            
            experience_stats = {}
            if 'experience' in clean_df.columns:
                non_null_experience = clean_df['experience'].notna().sum()
                experience_stats = {
                    'records_with_experience': int(non_null_experience),
                    'experience_extraction_rate': round(non_null_experience / len(clean_df) * 100, 2) if len(clean_df) > 0 else 0,
                    'avg_experience': round(clean_df['experience'].mean(), 2) if non_null_experience > 0 else None
                }
            
            summary[source] = {
                **basic_stats,
                'data_quality_score': quality_score,
                'validation_errors': len(validation_results['validation_errors']),
                'validation_warnings': len(validation_results['warnings']),
                'skills_extraction': skills_stats,
                'experience_extraction': experience_stats,
                'validation_details': validation_results
            }
        
        return summary
    
    def analyze_schema_compatibility(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze schema compatibility across data sources.
        
        Args:
            data: Dictionary of DataFrames from different sources
            
        Returns:
            Dict[str, Any]: Schema compatibility analysis
        """
        logger.info("Analyzing schema compatibility...")
        
        # Detect schemas for each source
        schemas = {}
        for source, df in data.items():
            if not df.empty:
                schemas[source] = self.schema_matcher.detect_schema(df, source)
        
        # Create unified schema
        unified_schema = self.schema_matcher.create_unified_schema(schemas)
        
        # Validate compatibility
        compatibility = self.schema_matcher.validate_schema_compatibility(schemas)
        
        return {
            'schemas': schemas,
            'unified_schema': unified_schema,
            'compatibility': compatibility
        }
    
    def perform_data_matching(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Perform data matching and entity resolution.
        
        Args:
            data: Dictionary of cleaned DataFrames
            
        Returns:
            Dict[str, Any]: Data matching results
        """
        logger.info("Performing data matching...")
        
        # Combine all data for matching analysis
        all_data = []
        for source, df in data.items():
            if not df.empty:
                df_copy = df.copy()
                df_copy['source'] = source
                all_data.append(df_copy)
        
        if not all_data:
            return {'error': 'No data available for matching'}
        
        # Filter out empty DataFrames before concatenation
        non_empty_data = [df for df in all_data if not df.empty and len(df) > 0]
        if non_empty_data:
            # Ensure all DataFrames have the same columns before concatenation
            all_columns = set()
            for df in non_empty_data:
                all_columns.update(df.columns)
            
            # Add missing columns to each DataFrame
            for df in non_empty_data:
                for col in all_columns:
                    if col not in df.columns:
                        df[col] = None
            
            combined_df = pd.concat(non_empty_data, ignore_index=True)
        else:
            combined_df = pd.DataFrame()
        
        # Perform matching analysis
        matching_report = self.data_matcher.generate_matching_report(combined_df)
        
        # Find duplicates and similar records
        df_with_duplicates = self.data_matcher.find_duplicates(combined_df)
        similar_groups = self.data_matcher.find_similar_records(combined_df)
        
        # Entity resolution
        entity_resolution = self.data_matcher.resolve_entities(combined_df)
        
        return {
            'matching_report': matching_report,
            'duplicates': df_with_duplicates,
            'similar_groups': similar_groups,
            'entity_resolution': entity_resolution,
            'combined_data': combined_df
        }
    
    def get_comprehensive_analysis(self, original_data: Dict[str, pd.DataFrame], 
                                 cleaned_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Get comprehensive analysis including schema matching and data matching.
        
        Args:
            original_data: Original data before cleaning
            cleaned_data: Data after cleaning
            
        Returns:
            Dict[str, Any]: Comprehensive analysis results
        """
        logger.info("Performing comprehensive analysis...")
        
        # Basic cleaning summary
        cleaning_summary = self.get_cleaning_summary(original_data, cleaned_data)
        
        # Schema compatibility analysis
        schema_analysis = self.analyze_schema_compatibility(cleaned_data)
        
        # Data matching analysis
        matching_analysis = self.perform_data_matching(cleaned_data)
        
        # Combine all results
        comprehensive_analysis = {
            'cleaning_summary': cleaning_summary,
            'schema_analysis': schema_analysis,
            'matching_analysis': matching_analysis,
            'overall_quality_score': self._calculate_overall_quality_score(
                cleaning_summary, schema_analysis, matching_analysis
            )
        }
        
        return comprehensive_analysis
    
    def _calculate_overall_quality_score(self, cleaning_summary: Dict, 
                                       schema_analysis: Dict, 
                                       matching_analysis: Dict) -> float:
        """
        Calculate overall data quality score.
        
        Args:
            cleaning_summary: Cleaning summary results
            schema_analysis: Schema analysis results
            matching_analysis: Matching analysis results
            
        Returns:
            float: Overall quality score (0-100)
        """
        scores = []
        
        # Data cleaning quality scores
        for source, stats in cleaning_summary.items():
            if 'data_quality_score' in stats:
                scores.append(stats['data_quality_score'])
        
        # Schema compatibility score
        if 'compatibility' in schema_analysis:
            compatibility_score = schema_analysis['compatibility']['overall_compatibility'] * 100
            scores.append(compatibility_score)
        
        # Data matching quality (inverse of duplicate/similarity rates)
        if 'matching_report' in matching_analysis:
            matching_report = matching_analysis['matching_report']
            duplicate_rate = matching_report['duplicate_analysis']['duplicate_rate']
            similarity_rate = matching_report['similarity_analysis']['similarity_rate']
            
            # Lower duplicate/similarity rates = higher quality
            matching_quality = (1 - (duplicate_rate + similarity_rate) / 2) * 100
            scores.append(matching_quality)
        
        return round(np.mean(scores), 2) if scores else 0.0
    
    def clean_and_store_data(self, data: Dict[str, pd.DataFrame], 
                           store_in_mongodb: bool = True) -> Dict[str, Any]:
        """
        Clean data and store in both PostgreSQL and MongoDB.
        
        Args:
            data: Dictionary of raw DataFrames
            store_in_mongodb: Whether to store in MongoDB
            
        Returns:
            Dict[str, Any]: Comprehensive results including storage info
        """
        logger.info("Starting comprehensive data cleaning and storage process...")
        
        # Step 1: Clean all data
        cleaned_data = self.clean_all_data(data)
        standardized_data = self.standardize_columns(cleaned_data)
        
        # Step 2: Get comprehensive analysis
        comprehensive_analysis = self.get_comprehensive_analysis(data, standardized_data)
        
        # Step 3: Store in MongoDB if enabled
        mongodb_results = {}
        if store_in_mongodb and self.mongodb_storage:
            try:
                logger.info("Storing data in MongoDB...")
                
                # Store raw data
                raw_stored = self.mongodb_storage.store_raw_job_data(data)
                
                # Store processed data
                processed_stored = self.mongodb_storage.store_processed_job_data(standardized_data)
                
                # Store job descriptions for full-text search
                descriptions_stored = self.mongodb_storage.store_job_descriptions(standardized_data)
                
                # Store analytics metadata
                analytics_stored = self.mongodb_storage.store_analytics_metadata({
                    'comprehensive_analysis': comprehensive_analysis
                })
                
                mongodb_results = {
                    'raw_data_stored': raw_stored,
                    'processed_data_stored': processed_stored,
                    'job_descriptions_stored': descriptions_stored,
                    'analytics_metadata_stored': analytics_stored,
                    'mongodb_summary': self.mongodb_storage.get_analytics_summary()
                }
                
                logger.info("MongoDB storage completed successfully")
                
            except Exception as e:
                logger.error(f"Failed to store data in MongoDB: {e}")
                mongodb_results = {'error': str(e)}
        else:
            mongodb_results = {'skipped': 'MongoDB storage not enabled or not available'}
        
        # Combine all results
        final_results = {
            'cleaning_analysis': comprehensive_analysis,
            'mongodb_storage': mongodb_results,
            'data_ready_for_postgresql': standardized_data,
            'processing_summary': {
                'total_sources': len(data),
                'total_records_processed': sum(len(df) for df in standardized_data.values()),
                'mongodb_enabled': store_in_mongodb and self.mongodb_storage is not None
            }
        }
        
        logger.info("Comprehensive data cleaning and storage process completed")
        return final_results
    
    def get_mongodb_search_capabilities(self) -> Dict[str, Any]:
        """
        Get MongoDB search capabilities summary.
        
        Returns:
            Dict[str, Any]: Search capabilities information
        """
        if not self.mongodb_storage:
            return {'error': 'MongoDB storage not available'}
        
        try:
            # Get collection counts
            summary = self.mongodb_storage.get_analytics_summary()
            
            # Test search capabilities
            search_capabilities = {
                'collections_available': list(summary.keys()),
                'total_documents': sum(v for v in summary.values() if isinstance(v, int)),
                'full_text_search_enabled': True,
                'skills_search_enabled': True,
                'company_search_enabled': True,
                'analytics_metadata_available': len(summary.get('latest_analytics', [])) > 0
            }
            
            return search_capabilities
            
        except Exception as e:
            logger.error(f"Failed to get MongoDB search capabilities: {e}")
            return {'error': str(e)}


if __name__ == "__main__":
    # Example usage
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    raw_data = loader.load_all_sources()
    
    # Clean data
    cleaner = DataCleaner()
    cleaned_data = cleaner.clean_all_data(raw_data)
    standardized_data = cleaner.standardize_columns(cleaned_data)
    
    # Print summary
    summary = cleaner.get_cleaning_summary(raw_data, standardized_data)
    for source, stats in summary.items():
        print(f"\n{source.upper()} Cleaning Summary:")
        print(f"  Original rows: {stats['original_rows']}")
        print(f"  Cleaned rows: {stats['cleaned_rows']}")
        print(f"  Rows removed: {stats['rows_removed']}")
        print(f"  Columns: {stats['columns_standardized']}")