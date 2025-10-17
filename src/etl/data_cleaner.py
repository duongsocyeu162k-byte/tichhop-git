"""
Data Cleaner Module
==================

Handles data cleaning, validation, and preprocessing for the job analytics project.
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import unicodedata

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    A class to handle data cleaning and preprocessing.
    """
    
    def __init__(self):
        """Initialize the DataCleaner."""
        self.cleaning_rules = {
            'remove_duplicates': True,
            'handle_missing_values': 'drop',
            'text_cleaning': True,
            'normalize_text': True
        }
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text data.
        
        Args:
            text: Input text to clean
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text) or text is None:
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep alphanumeric and basic punctuation
        text = re.sub(r'[^\w\s.,!?()-]', '', text)
        
        # Normalize unicode
        text = unicodedata.normalize('NFKD', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def extract_salary_range(self, salary_text: str) -> Tuple[Optional[int], Optional[int], str]:
        """
        Extract salary range from text.
        
        Args:
            salary_text: Salary text (e.g., "$50K-$70K", "50000-70000")
            
        Returns:
            Tuple[int, int, str]: (min_salary, max_salary, currency)
        """
        if pd.isna(salary_text) or not salary_text:
            return None, None, "USD"
        
        salary_text = str(salary_text).upper()
        
        # Extract currency
        currency = "USD"
        if "₹" in salary_text or "INR" in salary_text:
            currency = "INR"
        elif "€" in salary_text or "EUR" in salary_text:
            currency = "EUR"
        elif "£" in salary_text or "GBP" in salary_text:
            currency = "GBP"
        
        # Extract numbers
        numbers = re.findall(r'[\d,]+', salary_text.replace(',', ''))
        
        if len(numbers) >= 2:
            try:
                min_sal = int(numbers[0])
                max_sal = int(numbers[1])
                return min_sal, max_sal, currency
            except ValueError:
                pass
        
        elif len(numbers) == 1:
            try:
                salary = int(numbers[0])
                return salary, salary, currency
            except ValueError:
                pass
        
        return None, None, currency
    
    def extract_experience_years(self, experience_text: str) -> Optional[int]:
        """
        Extract years of experience from text.
        
        Args:
            experience_text: Experience text (e.g., "2-4 years", "5+ years")
            
        Returns:
            int: Years of experience (average if range)
        """
        if pd.isna(experience_text) or not experience_text:
            return None
        
        experience_text = str(experience_text).lower()
        
        # Extract numbers
        numbers = re.findall(r'\d+', experience_text)
        
        if numbers:
            if len(numbers) >= 2:
                # Range: take average
                return (int(numbers[0]) + int(numbers[1])) // 2
            else:
                return int(numbers[0])
        
        return None
    
    def clean_job_title(self, title: str) -> str:
        """
        Clean and standardize job titles.
        
        Args:
            title: Job title to clean
            
        Returns:
            str: Cleaned job title
        """
        if pd.isna(title) or not title:
            return ""
        
        title = str(title).strip()
        
        # Common title standardizations
        title_mappings = {
            'data scientist': 'Data Scientist',
            'data analyst': 'Data Analyst',
            'software engineer': 'Software Engineer',
            'software developer': 'Software Developer',
            'data engineer': 'Data Engineer',
            'business analyst': 'Business Analyst',
            'product manager': 'Product Manager',
            'project manager': 'Project Manager'
        }
        
        title_lower = title.lower()
        for key, value in title_mappings.items():
            if key in title_lower:
                return value
        
        return title
    
    def clean_location(self, location: str) -> Tuple[str, str]:
        """
        Clean and extract location information.
        
        Args:
            location: Location string
            
        Returns:
            Tuple[str, str]: (city, country)
        """
        if pd.isna(location) or not location:
            return "", ""
        
        location = str(location).strip()
        
        # Common country mappings
        country_mappings = {
            'usa': 'USA',
            'united states': 'USA',
            'us': 'USA',
            'india': 'India',
            'uk': 'UK',
            'united kingdom': 'UK',
            'canada': 'Canada',
            'australia': 'Australia'
        }
        
        # Extract city and country
        parts = location.split(',')
        if len(parts) >= 2:
            city = parts[0].strip()
            country = parts[-1].strip()
        else:
            city = location
            country = ""
        
        # Standardize country names
        country_lower = country.lower()
        for key, value in country_mappings.items():
            if key in country_lower:
                country = value
                break
        
        return city, country
    
    def clean_glassdoor_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean Glassdoor dataset.
        
        Args:
            df: Raw Glassdoor DataFrame
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        logger.info("Cleaning Glassdoor data...")
        
        # Create a copy
        cleaned_df = df.copy()
        
        # Clean text columns
        text_columns = ['Job Title', 'Job Description', 'Company Name', 'Location']
        for col in text_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].apply(self.clean_text)
        
        # Extract salary information
        if 'Salary Estimate' in cleaned_df.columns:
            salary_data = cleaned_df['Salary Estimate'].apply(self.extract_salary_range)
            cleaned_df['salary_min'] = [x[0] for x in salary_data]
            cleaned_df['salary_max'] = [x[1] for x in salary_data]
            cleaned_df['salary_currency'] = [x[2] for x in salary_data]
        
        # Clean job titles
        if 'Job Title' in cleaned_df.columns:
            cleaned_df['job_title_clean'] = cleaned_df['Job Title'].apply(self.clean_job_title)
        
        # Clean locations
        if 'Location' in cleaned_df.columns:
            location_data = cleaned_df['Location'].apply(self.clean_location)
            cleaned_df['city'] = [x[0] for x in location_data]
            cleaned_df['country'] = [x[1] for x in location_data]
        
        # Add source identifier
        cleaned_df['source'] = 'glassdoor'
        
        logger.info(f"Cleaned Glassdoor data: {len(cleaned_df)} rows")
        return cleaned_df
    
    def clean_monster_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean Monster.com dataset.
        
        Args:
            df: Raw Monster DataFrame
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        logger.info("Cleaning Monster data...")
        
        # Create a copy
        cleaned_df = df.copy()
        
        # Clean text columns
        text_columns = ['job_title', 'job_description', 'organization', 'location']
        for col in text_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].apply(self.clean_text)
        
        # Extract salary information
        if 'salary' in cleaned_df.columns:
            salary_data = cleaned_df['salary'].apply(self.extract_salary_range)
            cleaned_df['salary_min'] = [x[0] for x in salary_data]
            cleaned_df['salary_max'] = [x[1] for x in salary_data]
            cleaned_df['salary_currency'] = [x[2] for x in salary_data]
        
        # Clean job titles
        if 'job_title' in cleaned_df.columns:
            cleaned_df['job_title_clean'] = cleaned_df['job_title'].apply(self.clean_job_title)
        
        # Clean locations
        if 'location' in cleaned_df.columns:
            location_data = cleaned_df['location'].apply(self.clean_location)
            cleaned_df['city'] = [x[0] for x in location_data]
            cleaned_df['country'] = [x[1] for x in location_data]
        
        # Add source identifier
        cleaned_df['source'] = 'monster'
        
        logger.info(f"Cleaned Monster data: {len(cleaned_df)} rows")
        return cleaned_df
    
    def clean_naukri_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean Naukri.com dataset.
        
        Args:
            df: Raw Naukri DataFrame
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        logger.info("Cleaning Naukri data...")
        
        # Create a copy
        cleaned_df = df.copy()
        
        # Clean text columns
        text_columns = ['jobtitle', 'jobdescription', 'company', 'joblocation_address']
        for col in text_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].apply(self.clean_text)
        
        # Extract salary information
        if 'payrate' in cleaned_df.columns:
            salary_data = cleaned_df['payrate'].apply(self.extract_salary_range)
            cleaned_df['salary_min'] = [x[0] for x in salary_data]
            cleaned_df['salary_max'] = [x[1] for x in salary_data]
            cleaned_df['salary_currency'] = [x[2] for x in salary_data]
        
        # Extract experience
        if 'experience' in cleaned_df.columns:
            cleaned_df['experience_years'] = cleaned_df['experience'].apply(self.extract_experience_years)
        
        # Clean job titles
        if 'jobtitle' in cleaned_df.columns:
            cleaned_df['job_title_clean'] = cleaned_df['jobtitle'].apply(self.clean_job_title)
        
        # Clean locations
        if 'joblocation_address' in cleaned_df.columns:
            location_data = cleaned_df['joblocation_address'].apply(self.clean_location)
            cleaned_df['city'] = [x[0] for x in location_data]
            cleaned_df['country'] = [x[1] for x in location_data]
        
        # Add source identifier
        cleaned_df['source'] = 'naukri'
        
        logger.info(f"Cleaned Naukri data: {len(cleaned_df)} rows")
        return cleaned_df
    
    def clean_all_data(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Clean all datasets.
        
        Args:
            data_dict: Dictionary of raw DataFrames
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of cleaned DataFrames
        """
        cleaned_data = {}
        
        for source, df in data_dict.items():
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


if __name__ == "__main__":
    # Example usage
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    raw_data = loader.load_all_sources()
    
    # Clean data
    cleaner = DataCleaner()
    cleaned_data = cleaner.clean_all_data(raw_data)
    
    # Print summary
    for source, df in cleaned_data.items():
        print(f"\n{source.upper()} - Cleaned Data:")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
