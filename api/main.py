"""
FastAPI Backend for Job Market Analytics
========================================

REST API for accessing job market analytics data and insights.
"""

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
import sys
import os
from datetime import datetime
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from etl.data_loader import DataLoader
from etl.data_cleaner import DataCleaner
from analytics.trend_analyzer import TrendAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Job Market Analytics API",
    description="REST API for job market analytics and insights",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for data
data_loader = None
data_cleaner = None
trend_analyzer = None
all_data = None
cleaned_data = None

@app.on_event("startup")
async def startup_event():
    """Initialize data on startup."""
    global data_loader, data_cleaner, trend_analyzer, all_data, cleaned_data
    
    try:
        logger.info("Initializing data components...")
        
        # Initialize components
        data_loader = DataLoader()
        data_cleaner = DataCleaner()
        trend_analyzer = TrendAnalyzer()
        
        # Load and clean data
        logger.info("Loading raw data...")
        raw_data = data_loader.load_all_sources()
        
        logger.info("Cleaning data...")
        cleaned_data = data_cleaner.clean_all_data(raw_data)
        
        # Combine all data
        all_data = pd.concat(cleaned_data.values(), ignore_index=True)
        
        logger.info(f"Data loaded successfully: {len(all_data)} records")
        
    except Exception as e:
        logger.error(f"Error initializing data: {e}")
        all_data = pd.DataFrame()
        cleaned_data = {}

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Job Market Analytics API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "jobs": "/api/jobs",
            "analytics": "/api/analytics",
            "trends": "/api/trends",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "data_loaded": len(all_data) > 0 if all_data is not None else False
    }

@app.get("/api/jobs")
async def get_jobs(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    source: Optional[str] = Query(None),
    job_title: Optional[str] = Query(None),
    country: Optional[str] = Query(None),
    min_salary: Optional[int] = Query(None),
    max_salary: Optional[int] = Query(None)
):
    """
    Get job listings with optional filters.
    
    Args:
        limit: Maximum number of results
        offset: Number of results to skip
        source: Filter by data source (glassdoor, monster, naukri)
        job_title: Filter by job title
        country: Filter by country
        min_salary: Minimum salary filter
        max_salary: Maximum salary filter
    """
    try:
        if all_data is None or all_data.empty:
            raise HTTPException(status_code=404, detail="No data available")
        
        # Apply filters
        filtered_data = all_data.copy()
        
        if source:
            filtered_data = filtered_data[filtered_data['source'] == source]
        
        if job_title:
            filtered_data = filtered_data[
                filtered_data['job_title_clean'].str.contains(job_title, case=False, na=False)
            ]
        
        if country:
            filtered_data = filtered_data[filtered_data['country'] == country]
        
        if min_salary is not None:
            filtered_data = filtered_data[filtered_data['salary_min'] >= min_salary]
        
        if max_salary is not None:
            filtered_data = filtered_data[filtered_data['salary_max'] <= max_salary]
        
        # Apply pagination
        total_count = len(filtered_data)
        paginated_data = filtered_data.iloc[offset:offset + limit]
        
        return {
            "data": paginated_data.to_dict('records'),
            "pagination": {
                "total": total_count,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total_count
            }
        }
        
    except Exception as e:
        logger.error(f"Error in get_jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/summary")
async def get_analytics_summary():
    """Get analytics summary."""
    try:
        if all_data is None or all_data.empty:
            raise HTTPException(status_code=404, detail="No data available")
        
        # Basic statistics
        total_jobs = len(all_data)
        unique_companies = all_data['company_name'].nunique() if 'company_name' in all_data.columns else 0
        unique_locations = all_data['city'].nunique() if 'city' in all_data.columns else 0
        
        # Source distribution
        source_distribution = all_data['source'].value_counts().to_dict() if 'source' in all_data.columns else {}
        
        # Top job titles
        top_job_titles = all_data['job_title_clean'].value_counts().head(10).to_dict() if 'job_title_clean' in all_data.columns else {}
        
        # Top countries
        top_countries = all_data['country'].value_counts().head(10).to_dict() if 'country' in all_data.columns else {}
        
        # Salary statistics
        salary_stats = {}
        if 'salary_min' in all_data.columns and 'salary_max' in all_data.columns:
            all_data['avg_salary'] = (all_data['salary_min'] + all_data['salary_max']) / 2
            salary_stats = {
                'mean': all_data['avg_salary'].mean(),
                'median': all_data['avg_salary'].median(),
                'min': all_data['avg_salary'].min(),
                'max': all_data['avg_salary'].max()
            }
        
        return {
            "summary": {
                "total_jobs": total_jobs,
                "unique_companies": unique_companies,
                "unique_locations": unique_locations
            },
            "source_distribution": source_distribution,
            "top_job_titles": top_job_titles,
            "top_countries": top_countries,
            "salary_statistics": salary_stats
        }
        
    except Exception as e:
        logger.error(f"Error in get_analytics_summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/trends")
async def get_trends():
    """Get trend analysis."""
    try:
        if all_data is None or all_data.empty:
            raise HTTPException(status_code=404, detail="No data available")
        
        if trend_analyzer is None:
            raise HTTPException(status_code=500, detail="Trend analyzer not initialized")
        
        # Generate trend report
        trend_report = trend_analyzer.generate_trend_report(all_data)
        
        return trend_report
        
    except Exception as e:
        logger.error(f"Error in get_trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/salary-prediction")
async def predict_salary(
    job_title: str,
    location: Optional[str] = None,
    experience_years: Optional[int] = None,
    industry: Optional[str] = None
):
    """
    Predict salary based on job characteristics.
    
    Args:
        job_title: Job title
        location: Job location
        experience_years: Years of experience
        industry: Industry
    """
    try:
        if all_data is None or all_data.empty:
            raise HTTPException(status_code=404, detail="No data available")
        
        # Filter data based on criteria
        filtered_data = all_data.copy()
        
        if job_title:
            filtered_data = filtered_data[
                filtered_data['job_title_clean'].str.contains(job_title, case=False, na=False)
            ]
        
        if location:
            filtered_data = filtered_data[
                filtered_data['city'].str.contains(location, case=False, na=False)
            ]
        
        if industry:
            filtered_data = filtered_data[
                filtered_data['industry'].str.contains(industry, case=False, na=False)
            ]
        
        if len(filtered_data) == 0:
            return {
                "prediction": None,
                "message": "No matching data found for prediction",
                "sample_size": 0
            }
        
        # Calculate salary statistics
        if 'salary_min' in filtered_data.columns and 'salary_max' in filtered_data.columns:
            filtered_data['avg_salary'] = (filtered_data['salary_min'] + filtered_data['salary_max']) / 2
            
            salary_stats = {
                'predicted_min': filtered_data['salary_min'].mean(),
                'predicted_max': filtered_data['salary_max'].mean(),
                'predicted_avg': filtered_data['avg_salary'].mean(),
                'median': filtered_data['avg_salary'].median(),
                'std': filtered_data['avg_salary'].std()
            }
            
            return {
                "prediction": salary_stats,
                "sample_size": len(filtered_data),
                "criteria": {
                    "job_title": job_title,
                    "location": location,
                    "experience_years": experience_years,
                    "industry": industry
                }
            }
        else:
            return {
                "prediction": None,
                "message": "Salary data not available",
                "sample_size": len(filtered_data)
            }
        
    except Exception as e:
        logger.error(f"Error in predict_salary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/skills")
async def get_skills_analysis():
    """Get skills analysis."""
    try:
        if all_data is None or all_data.empty:
            raise HTTPException(status_code=404, detail="No data available")
        
        # Extract skills from all sources
        all_skills = []
        for skills_str in all_data['skills'].dropna():
            if isinstance(skills_str, str):
                skills = [skill.strip() for skill in skills_str.split(',')]
                all_skills.extend(skills)
        
        # Count skills
        from collections import Counter
        skills_counter = Counter(all_skills)
        
        # Get top skills
        top_skills = dict(skills_counter.most_common(20))
        
        return {
            "total_unique_skills": len(skills_counter),
            "top_skills": top_skills,
            "skills_distribution": dict(skills_counter)
        }
        
    except Exception as e:
        logger.error(f"Error in get_skills_analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/geographic")
async def get_geographic_analysis():
    """Get geographic analysis."""
    try:
        if all_data is None or all_data.empty:
            raise HTTPException(status_code=404, detail="No data available")
        
        # Country analysis
        country_stats = {}
        if 'country' in all_data.columns:
            country_counts = all_data['country'].value_counts()
            country_stats = {
                'distribution': country_counts.to_dict(),
                'top_countries': country_counts.head(10).to_dict()
            }
        
        # City analysis
        city_stats = {}
        if 'city' in all_data.columns:
            city_counts = all_data['city'].value_counts()
            city_stats = {
                'distribution': city_counts.to_dict(),
                'top_cities': city_counts.head(20).to_dict()
            }
        
        return {
            "countries": country_stats,
            "cities": city_stats
        }
        
    except Exception as e:
        logger.error(f"Error in get_geographic_analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
