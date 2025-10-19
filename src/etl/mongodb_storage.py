"""
MongoDB Storage Module
=====================

Handles MongoDB operations for storing semi-structured and unstructured data
in the Job Market Analytics system.
"""

import pymongo
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
from bson import ObjectId
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, DuplicateKeyError

logger = logging.getLogger(__name__)


class MongoDBStorage:
    """
    A class to handle MongoDB operations for job market analytics data.
    """
    
    def __init__(self, connection_string: str = "mongodb://admin:password123@localhost:27017/job_analytics"):
        """
        Initialize MongoDB connection.
        
        Args:
            connection_string: MongoDB connection string
        """
        self.connection_string = connection_string
        self.client = None
        self.db = None
        self._connect()
        
        # Collection names
        self.collections = {
            'raw_jobs': 'raw_job_postings',
            'processed_jobs': 'processed_job_postings', 
            'company_profiles': 'company_profiles',
            'skills_data': 'skills_data',
            'analytics_metadata': 'analytics_metadata',
            'job_descriptions': 'job_descriptions',
            'market_trends': 'market_trends'
        }
        
        # Create indexes for better performance
        self._create_indexes()
    
    def _connect(self):
        """Establish MongoDB connection."""
        try:
            self.client = MongoClient(self.connection_string, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client.job_analytics
            logger.info("Successfully connected to MongoDB")
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def _create_indexes(self):
        """Create indexes for better query performance."""
        try:
            # Indexes for raw_job_postings
            self.db[self.collections['raw_jobs']].create_index([
                ("source", ASCENDING),
                ("job_title", ASCENDING)
            ])
            self.db[self.collections['raw_jobs']].create_index("company_name")
            self.db[self.collections['raw_jobs']].create_index("location")
            
            # Indexes for processed_job_postings
            self.db[self.collections['processed_jobs']].create_index([
                ("source", ASCENDING),
                ("job_title_clean", ASCENDING)
            ])
            self.db[self.collections['processed_jobs']].create_index("company_name")
            self.db[self.collections['processed_jobs']].create_index("country")
            self.db[self.collections['processed_jobs']].create_index("industry")
            
            # Indexes for company_profiles
            self.db[self.collections['company_profiles']].create_index("company_name", unique=True)
            self.db[self.collections['company_profiles']].create_index("industry")
            
            # Indexes for skills_data
            self.db[self.collections['skills_data']].create_index("skill_name")
            self.db[self.collections['skills_data']].create_index("category")
            
            # Text index for job descriptions
            self.db[self.collections['job_descriptions']].create_index([
                ("job_description", "text"),
                ("job_title", "text")
            ])
            
            logger.info("MongoDB indexes created successfully")
        except Exception as e:
            logger.warning(f"Failed to create some indexes: {e}")
    
    def store_raw_job_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, int]:
        """
        Store raw job data from all sources.
        
        Args:
            data: Dictionary of raw DataFrames from different sources
            
        Returns:
            Dict[str, int]: Number of records stored per source
        """
        logger.info("Storing raw job data to MongoDB...")
        stored_counts = {}
        
        for source, df in data.items():
            if df.empty:
                stored_counts[source] = 0
                continue
            
            try:
                # Convert DataFrame to list of dictionaries
                records = df.to_dict('records')
                
                # Add metadata
                for record in records:
                    record['_source'] = source
                    record['_created_at'] = datetime.utcnow()
                    record['_data_type'] = 'raw'
                
                # Insert into MongoDB
                collection = self.db[self.collections['raw_jobs']]
                result = collection.insert_many(records)
                stored_counts[source] = len(result.inserted_ids)
                
                logger.info(f"Stored {stored_counts[source]} raw records from {source}")
                
            except Exception as e:
                logger.error(f"Failed to store raw data from {source}: {e}")
                stored_counts[source] = 0
        
        return stored_counts
    
    def store_processed_job_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, int]:
        """
        Store processed job data with enhanced structure.
        
        Args:
            data: Dictionary of processed DataFrames
            
        Returns:
            Dict[str, int]: Number of records stored per source
        """
        logger.info("Storing processed job data to MongoDB...")
        stored_counts = {}
        
        for source, df in data.items():
            if df.empty:
                stored_counts[source] = 0
                continue
            
            try:
                # Convert DataFrame to list of dictionaries
                records = df.to_dict('records')
                
                # Enhance records with MongoDB-specific structure
                enhanced_records = []
                for record in records:
                    enhanced_record = self._enhance_record_for_mongodb(record, source)
                    enhanced_records.append(enhanced_record)
                
                # Insert into MongoDB
                collection = self.db[self.collections['processed_jobs']]
                result = collection.insert_many(enhanced_records)
                stored_counts[source] = len(result.inserted_ids)
                
                logger.info(f"Stored {stored_counts[source]} processed records from {source}")
                
            except Exception as e:
                logger.error(f"Failed to store processed data from {source}: {e}")
                stored_counts[source] = 0
        
        return stored_counts
    
    def _enhance_record_for_mongodb(self, record: Dict, source: str) -> Dict:
        """
        Enhance record with MongoDB-specific structure.
        
        Args:
            record: Original record
            source: Data source
            
        Returns:
            Dict: Enhanced record
        """
        enhanced = record.copy()
        
        # Add metadata
        enhanced['_source'] = source
        enhanced['_created_at'] = datetime.utcnow()
        enhanced['_data_type'] = 'processed'
        enhanced['_version'] = '1.0'
        
        # Structure skills as array
        if 'skills' in enhanced and enhanced['skills']:
            skills_str = str(enhanced['skills'])
            if skills_str and skills_str != 'nan':
                enhanced['skills_array'] = [skill.strip() for skill in skills_str.split(',') if skill.strip()]
            else:
                enhanced['skills_array'] = []
        else:
            enhanced['skills_array'] = []
        
        # Structure location information
        location_info = {
            'raw': enhanced.get('location_clean', ''),
            'city': enhanced.get('city', ''),
            'state': enhanced.get('state', ''),
            'country': enhanced.get('country', ''),
            'coordinates': None  # Can be enhanced with geocoding later
        }
        enhanced['location'] = location_info
        
        # Structure salary information
        salary_info = {
            'min': enhanced.get('salary_min'),
            'max': enhanced.get('salary_max'),
            'currency': 'USD',  # Default, can be enhanced
            'period': 'yearly'  # Default, can be enhanced
        }
        enhanced['salary'] = salary_info
        
        # Structure company information
        company_info = {
            'name': enhanced.get('company_name', ''),
            'size': enhanced.get('company_size', ''),
            'industry': enhanced.get('industry', ''),
            'rating': enhanced.get('rating')
        }
        enhanced['company'] = company_info
        
        # Structure job information
        job_info = {
            'title': enhanced.get('job_title_clean', ''),
            'description': enhanced.get('job_description', ''),
            'experience_required': enhanced.get('experience'),
            'skills_required': enhanced['skills_array']
        }
        enhanced['job'] = job_info
        
        return enhanced
    
    def store_job_descriptions(self, data: Dict[str, pd.DataFrame]) -> int:
        """
        Store job descriptions separately for full-text search.
        
        Args:
            data: Dictionary of DataFrames with job descriptions
            
        Returns:
            int: Total number of job descriptions stored
        """
        logger.info("Storing job descriptions for full-text search...")
        total_stored = 0
        
        for source, df in data.items():
            if df.empty or 'job_description' not in df.columns:
                continue
            
            try:
                descriptions = []
                for idx, row in df.iterrows():
                    if pd.notna(row['job_description']) and str(row['job_description']).strip():
                        description_doc = {
                            '_source': source,
                            'job_title': row.get('job_title_clean', ''),
                            'company_name': row.get('company_name', ''),
                            'job_description': str(row['job_description']),
                            'skills': row.get('skills', ''),
                            'location': row.get('location_clean', ''),
                            'created_at': datetime.utcnow()
                        }
                        descriptions.append(description_doc)
                
                if descriptions:
                    collection = self.db[self.collections['job_descriptions']]
                    result = collection.insert_many(descriptions)
                    total_stored += len(result.inserted_ids)
                    logger.info(f"Stored {len(result.inserted_ids)} job descriptions from {source}")
                
            except Exception as e:
                logger.error(f"Failed to store job descriptions from {source}: {e}")
        
        return total_stored
    
    def store_skills_data(self, skills_analysis: Dict[str, Any]) -> int:
        """
        Store skills analysis data.
        
        Args:
            skills_analysis: Skills analysis results
            
        Returns:
            int: Number of skills records stored
        """
        logger.info("Storing skills analysis data...")
        
        try:
            collection = self.db[self.collections['skills_data']]
            stored_count = 0
            
            # Store individual skills
            if 'top_skills' in skills_analysis:
                for skill_data in skills_analysis['top_skills']:
                    skill_doc = {
                        'skill_name': skill_data.get('skill', ''),
                        'frequency': skill_data.get('count', 0),
                        'percentage': skill_data.get('percentage', 0),
                        'category': self._categorize_skill(skill_data.get('skill', '')),
                        'created_at': datetime.utcnow()
                    }
                    collection.insert_one(skill_doc)
                    stored_count += 1
            
            # Store skills by category
            if 'skills_by_category' in skills_analysis:
                for category, skills in skills_analysis['skills_by_category'].items():
                    category_doc = {
                        'category': category,
                        'skills': skills,
                        'skill_count': len(skills),
                        'created_at': datetime.utcnow()
                    }
                    collection.insert_one(category_doc)
                    stored_count += 1
            
            logger.info(f"Stored {stored_count} skills records")
            return stored_count
            
        except Exception as e:
            logger.error(f"Failed to store skills data: {e}")
            return 0
    
    def _categorize_skill(self, skill: str) -> str:
        """
        Categorize skill into predefined categories.
        
        Args:
            skill: Skill name
            
        Returns:
            str: Skill category
        """
        skill_lower = skill.lower()
        
        if any(tech in skill_lower for tech in ['python', 'java', 'javascript', 'sql', 'r', 'scala', 'go', 'c++', 'c#']):
            return 'Programming Languages'
        elif any(tech in skill_lower for tech in ['machine learning', 'ml', 'deep learning', 'data science', 'analytics']):
            return 'Data Science & ML'
        elif any(tech in skill_lower for tech in ['tableau', 'power bi', 'excel', 'matplotlib', 'seaborn']):
            return 'Visualization & BI'
        elif any(tech in skill_lower for tech in ['aws', 'azure', 'gcp', 'docker', 'kubernetes']):
            return 'Cloud & DevOps'
        elif any(tech in skill_lower for tech in ['mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch']):
            return 'Databases'
        elif any(tech in skill_lower for tech in ['html', 'css', 'react', 'angular', 'vue', 'node.js']):
            return 'Web Technologies'
        else:
            return 'Other'
    
    def store_analytics_metadata(self, analytics_results: Dict[str, Any]) -> int:
        """
        Store analytics metadata and results.
        
        Args:
            analytics_results: Analytics results from trend analyzer
            
        Returns:
            int: Number of metadata records stored
        """
        logger.info("Storing analytics metadata...")
        
        try:
            collection = self.db[self.collections['analytics_metadata']]
            stored_count = 0
            
            # Store each analytics result as a separate document
            for analysis_type, results in analytics_results.items():
                metadata_doc = {
                    'analysis_type': analysis_type,
                    'results': results,
                    'created_at': datetime.utcnow(),
                    'data_version': '1.0'
                }
                collection.insert_one(metadata_doc)
                stored_count += 1
            
            logger.info(f"Stored {stored_count} analytics metadata records")
            return stored_count
            
        except Exception as e:
            logger.error(f"Failed to store analytics metadata: {e}")
            return 0
    
    def search_jobs_by_text(self, search_text: str, limit: int = 10) -> List[Dict]:
        """
        Search jobs using full-text search.
        
        Args:
            search_text: Text to search for
            limit: Maximum number of results
            
        Returns:
            List[Dict]: Search results
        """
        try:
            collection = self.db[self.collections['job_descriptions']]
            
            # Perform text search
            results = collection.find(
                {"$text": {"$search": search_text}},
                {"score": {"$meta": "textScore"}}
            ).sort([("score", {"$meta": "textScore"})]).limit(limit)
            
            return list(results)
            
        except Exception as e:
            logger.error(f"Failed to search jobs: {e}")
            return []
    
    def get_jobs_by_skills(self, skills: List[str], limit: int = 10) -> List[Dict]:
        """
        Get jobs that require specific skills.
        
        Args:
            skills: List of skills to search for
            limit: Maximum number of results
            
        Returns:
            List[Dict]: Jobs matching the skills
        """
        try:
            collection = self.db[self.collections['processed_jobs']]
            
            # Search for jobs containing any of the specified skills
            query = {"skills_array": {"$in": skills}}
            results = collection.find(query).limit(limit)
            
            return list(results)
            
        except Exception as e:
            logger.error(f"Failed to get jobs by skills: {e}")
            return []
    
    def get_company_profiles(self, company_name: Optional[str] = None) -> List[Dict]:
        """
        Get company profiles.
        
        Args:
            company_name: Specific company name (optional)
            
        Returns:
            List[Dict]: Company profiles
        """
        try:
            collection = self.db[self.collections['company_profiles']]
            
            if company_name:
                query = {"company_name": {"$regex": company_name, "$options": "i"}}
            else:
                query = {}
            
            results = collection.find(query)
            return list(results)
            
        except Exception as e:
            logger.error(f"Failed to get company profiles: {e}")
            return []
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """
        Get summary of all analytics data.
        
        Returns:
            Dict[str, Any]: Analytics summary
        """
        try:
            summary = {}
            
            # Count documents in each collection
            for collection_name, collection_key in self.collections.items():
                count = self.db[collection_key].count_documents({})
                summary[collection_name] = count
            
            # Get latest analytics metadata
            analytics_collection = self.db[self.collections['analytics_metadata']]
            latest_analytics = analytics_collection.find().sort("created_at", -1).limit(1)
            summary['latest_analytics'] = list(latest_analytics)
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get analytics summary: {e}")
            return {}
    
    def close_connection(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Initialize MongoDB storage
    mongo_storage = MongoDBStorage()
    
    # Example data
    sample_data = {
        'glassdoor': pd.DataFrame({
            'job_title_clean': ['Data Scientist', 'Software Engineer'],
            'company_name': ['Google', 'Microsoft'],
            'job_description': ['Analyze data', 'Develop software'],
            'skills': ['python,machine learning', 'java,spring']
        })
    }
    
    # Store data
    stored_counts = mongo_storage.store_processed_job_data(sample_data)
    print(f"Stored {stored_counts} records")
    
    # Search example
    search_results = mongo_storage.search_jobs_by_text("data scientist")
    print(f"Found {len(search_results)} jobs")
    
    # Close connection
    mongo_storage.close_connection()
