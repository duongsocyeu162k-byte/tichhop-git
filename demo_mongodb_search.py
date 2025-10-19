#!/usr/bin/env python3
"""
MongoDB Search Demo
==================

Demo script to showcase MongoDB search capabilities after running the pipeline.
"""

import sys
import os
import logging
from typing import List, Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from etl.mongodb_storage import MongoDBStorage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_full_text_search(mongo_storage: MongoDBStorage):
    """Demo full-text search capabilities."""
    print("\n🔍 Full-Text Search Demo")
    print("=" * 30)
    
    # Search for data scientist jobs
    search_terms = ["data scientist", "machine learning", "python developer", "software engineer"]
    
    for term in search_terms:
        print(f"\nSearching for: '{term}'")
        results = mongo_storage.search_jobs_by_text(term, limit=3)
        
        if results:
            print(f"Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result.get('job_title', 'N/A')} at {result.get('company_name', 'N/A')}")
                print(f"     Location: {result.get('location', 'N/A')}")
                print(f"     Score: {result.get('score', 'N/A')}")
        else:
            print("  No results found")


def demo_skills_search(mongo_storage: MongoDBStorage):
    """Demo skills-based search."""
    print("\n🛠️  Skills-Based Search Demo")
    print("=" * 30)
    
    # Search for jobs with specific skills
    skill_sets = [
        ["python", "machine learning"],
        ["java", "spring"],
        ["javascript", "react"],
        ["sql", "data analysis"]
    ]
    
    for skills in skill_sets:
        print(f"\nSearching for jobs with skills: {', '.join(skills)}")
        results = mongo_storage.get_jobs_by_skills(skills, limit=3)
        
        if results:
            print(f"Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                job_info = result.get('job', {})
                company_info = result.get('company', {})
                print(f"  {i}. {job_info.get('title', 'N/A')} at {company_info.get('name', 'N/A')}")
                print(f"     Skills: {', '.join(result.get('skills_array', [])[:5])}")
                print(f"     Location: {result.get('location', {}).get('city', 'N/A')}")
        else:
            print("  No results found")


def demo_company_search(mongo_storage: MongoDBStorage):
    """Demo company-based search."""
    print("\n🏢 Company Search Demo")
    print("=" * 30)
    
    # Search for specific companies
    companies = ["Google", "Microsoft", "Amazon", "Apple", "Meta"]
    
    for company in companies:
        print(f"\nSearching for: {company}")
        results = mongo_storage.get_company_profiles(company)
        
        if results:
            print(f"Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result.get('company_name', 'N/A')}")
                print(f"     Industry: {result.get('industry', 'N/A')}")
                print(f"     Size: {result.get('company_size', 'N/A')}")
        else:
            print("  No results found")


def demo_analytics_summary(mongo_storage: MongoDBStorage):
    """Demo analytics summary."""
    print("\n📊 Analytics Summary Demo")
    print("=" * 30)
    
    summary = mongo_storage.get_analytics_summary()
    
    if summary:
        print("MongoDB Collections Summary:")
        for collection, count in summary.items():
            if isinstance(count, int):
                print(f"  📁 {collection}: {count:,} documents")
        
        # Show latest analytics
        if 'latest_analytics' in summary and summary['latest_analytics']:
            print(f"\nLatest Analytics Metadata: {len(summary['latest_analytics'])} records")
    else:
        print("No analytics summary available")


def demo_advanced_queries(mongo_storage: MongoDBStorage):
    """Demo advanced MongoDB queries."""
    print("\n🚀 Advanced Queries Demo")
    print("=" * 30)
    
    try:
        # Query 1: Jobs with high salary
        print("\n1. High-salary jobs (>$100k):")
        high_salary_query = {
            "$or": [
                {"salary.min": {"$gte": 100000}},
                {"salary.max": {"$gte": 100000}}
            ]
        }
        
        collection = mongo_storage.db[mongo_storage.collections['processed_jobs']]
        high_salary_jobs = list(collection.find(high_salary_query).limit(3))
        
        if high_salary_jobs:
            for i, job in enumerate(high_salary_jobs, 1):
                job_info = job.get('job', {})
                salary_info = job.get('salary', {})
                print(f"  {i}. {job_info.get('title', 'N/A')}")
                print(f"     Salary: ${salary_info.get('min', 'N/A')} - ${salary_info.get('max', 'N/A')}")
        else:
            print("  No high-salary jobs found")
        
        # Query 2: Jobs by location
        print("\n2. Jobs in major cities:")
        major_cities = ["San Francisco", "New York", "Seattle", "Austin", "Boston"]
        
        for city in major_cities:
            city_query = {"location.city": {"$regex": city, "$options": "i"}}
            city_jobs = list(collection.find(city_query).limit(1))
            
            if city_jobs:
                job = city_jobs[0]
                job_info = job.get('job', {})
                print(f"  📍 {city}: {job_info.get('title', 'N/A')} at {job.get('company', {}).get('name', 'N/A')}")
        
        # Query 3: Skills aggregation
        print("\n3. Most common skills:")
        pipeline = [
            {"$unwind": "$skills_array"},
            {"$group": {"_id": "$skills_array", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 10}
        ]
        
        skills_aggregation = list(collection.aggregate(pipeline))
        if skills_aggregation:
            for i, skill in enumerate(skills_aggregation, 1):
                print(f"  {i}. {skill['_id']}: {skill['count']} jobs")
        
    except Exception as e:
        print(f"Error in advanced queries: {e}")


def main():
    """Main demo function."""
    print("🚀 MongoDB Search Capabilities Demo")
    print("=" * 50)
    
    try:
        # Initialize MongoDB storage
        mongo_storage = MongoDBStorage()
        
        # Check if data exists
        summary = mongo_storage.get_analytics_summary()
        total_docs = sum(v for v in summary.values() if isinstance(v, int))
        
        if total_docs == 0:
            print("❌ No data found in MongoDB!")
            print("Please run 'python database_analytics_pipeline.py' first to populate the database.")
            return
        
        print(f"✅ Found {total_docs:,} documents in MongoDB")
        print("Starting search demos...")
        
        # Run all demos
        demo_full_text_search(mongo_storage)
        demo_skills_search(mongo_storage)
        demo_company_search(mongo_storage)
        demo_analytics_summary(mongo_storage)
        demo_advanced_queries(mongo_storage)
        
        print("\n🎉 All demos completed successfully!")
        print("\n💡 Try these MongoDB commands:")
        print("   docker compose exec mongodb mongosh -u admin -p password123")
        print("   use job_analytics")
        print("   db.processed_job_postings.find().limit(5)")
        print("   db.job_descriptions.find({\"$text\": {\"$search\": \"data scientist\"}})")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'mongo_storage' in locals():
            mongo_storage.close_connection()


if __name__ == "__main__":
    main()
