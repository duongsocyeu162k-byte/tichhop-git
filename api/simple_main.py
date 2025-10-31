"""
FastAPI Backend cho Job Market Analytics (Simplified Version)
===============================================================

REST API đơn giản để truy cập dữ liệu phân tích từ file JSON.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
import os
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import glob

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Job Market Analytics API",
    description="REST API đơn giản để truy cập dữ liệu phân tích việc làm",
    version="2.0.0",
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

# Global variables
OUTPUT_DIR = "output"
all_data = None
analytics_report = None
last_loaded = None


def load_latest_json_files():
    """Load dữ liệu từ các file JSON mới nhất."""
    global all_data, analytics_report, last_loaded
    
    try:
        # Tìm file processed_jobs mới nhất
        processed_files = glob.glob(os.path.join(OUTPUT_DIR, "processed_jobs_*.json"))
        if not processed_files:
            logger.warning("Không tìm thấy file processed_jobs")
            return False
        
        latest_processed = max(processed_files, key=os.path.getctime)
        
        # Tìm file analytics_report mới nhất
        analytics_files = glob.glob(os.path.join(OUTPUT_DIR, "analytics_report_*.json"))
        latest_analytics = max(analytics_files, key=os.path.getctime) if analytics_files else None
        
        # Load processed data
        logger.info(f"Đang tải dữ liệu từ {latest_processed}...")
        with open(latest_processed, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
            all_data = pd.DataFrame(data_list)
        
        # Load analytics report nếu có
        if latest_analytics:
            logger.info(f"Đang tải báo cáo phân tích từ {latest_analytics}...")
            with open(latest_analytics, 'r', encoding='utf-8') as f:
                analytics_report = json.load(f)
        
        last_loaded = datetime.now()
        logger.info(f"Đã tải {len(all_data)} records thành công")
        return True
        
    except Exception as e:
        logger.error(f"Lỗi khi tải file JSON: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    """Load dữ liệu khi khởi động."""
    logger.info("Khởi động API...")
    success = load_latest_json_files()
    if not success:
        logger.warning("Không thể tải dữ liệu ban đầu. API sẽ chạy mà không có dữ liệu.")


@app.get("/")
async def root():
    """Root endpoint với thông tin API."""
    return {
        "message": "Job Market Analytics API (JSON-based)",
        "version": "2.0.0",
        "status": "running",
        "data_loaded": all_data is not None and not all_data.empty,
        "total_records": len(all_data) if all_data is not None else 0,
        "last_loaded": last_loaded.isoformat() if last_loaded else None,
        "endpoints": {
            "jobs": "/api/jobs",
            "analytics": "/api/analytics/summary",
            "trends": "/api/analytics/trends",
            "salary_prediction": "/api/analytics/salary-prediction",
            "skills": "/api/analytics/skills",
            "geographic": "/api/analytics/geographic",
            "reload": "/api/reload",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "data_loaded": all_data is not None and not all_data.empty,
        "total_records": len(all_data) if all_data is not None else 0
    }


@app.post("/api/reload")
async def reload_data():
    """Reload dữ liệu từ file JSON."""
    success = load_latest_json_files()
    if success:
        return {
            "status": "success",
            "message": "Đã tải lại dữ liệu thành công",
            "total_records": len(all_data) if all_data is not None else 0,
            "timestamp": datetime.now().isoformat()
        }
    else:
        raise HTTPException(status_code=500, detail="Không thể tải lại dữ liệu")


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
    Lấy danh sách công việc với filters tùy chọn.
    
    Args:
        limit: Số lượng kết quả tối đa
        offset: Số lượng kết quả bỏ qua
        source: Lọc theo nguồn dữ liệu
        job_title: Lọc theo tên công việc
        country: Lọc theo quốc gia
        min_salary: Lương tối thiểu
        max_salary: Lương tối đa
    """
    try:
        if all_data is None or all_data.empty:
            raise HTTPException(status_code=404, detail="Không có dữ liệu")
        
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
            filtered_data = filtered_data[
                pd.to_numeric(filtered_data['salary_min'], errors='coerce') >= min_salary
            ]
        
        if max_salary is not None:
            filtered_data = filtered_data[
                pd.to_numeric(filtered_data['salary_max'], errors='coerce') <= max_salary
            ]
        
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
        logger.error(f"Lỗi trong get_jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/summary")
async def get_analytics_summary():
    """Lấy tóm tắt phân tích."""
    try:
        if all_data is None or all_data.empty:
            raise HTTPException(status_code=404, detail="Không có dữ liệu")
        
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
            salary_min_numeric = pd.to_numeric(all_data['salary_min'], errors='coerce')
            salary_max_numeric = pd.to_numeric(all_data['salary_max'], errors='coerce')
            avg_salary = (salary_min_numeric + salary_max_numeric) / 2
            
            salary_stats = {
                'mean': float(avg_salary.mean()) if not avg_salary.isna().all() else None,
                'median': float(avg_salary.median()) if not avg_salary.isna().all() else None,
                'min': float(avg_salary.min()) if not avg_salary.isna().all() else None,
                'max': float(avg_salary.max()) if not avg_salary.isna().all() else None
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
        logger.error(f"Lỗi trong get_analytics_summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/trends")
async def get_trends():
    """Lấy phân tích xu hướng từ analytics report."""
    try:
        if analytics_report is None:
            raise HTTPException(
                status_code=404, 
                detail="Báo cáo phân tích chưa có. Hãy chạy pipeline trước."
            )
        
        # Trả về phần trend analysis từ report
        trends_data = {}
        
        if 'trend_analysis' in analytics_report:
            trends_data = analytics_report['trend_analysis']
        elif 'market_trends' in analytics_report:
            trends_data = analytics_report['market_trends']
        
        return trends_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lỗi trong get_trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/salary-prediction")
async def predict_salary(
    job_title: str,
    location: Optional[str] = None,
    experience_years: Optional[int] = None,
    industry: Optional[str] = None
):
    """
    Dự đoán lương dựa trên đặc điểm công việc.
    
    Args:
        job_title: Tên công việc
        location: Địa điểm
        experience_years: Số năm kinh nghiệm
        industry: Ngành nghề
    """
    try:
        if all_data is None or all_data.empty:
            raise HTTPException(status_code=404, detail="Không có dữ liệu")
        
        # Filter data
        filtered_data = all_data.copy()
        
        if job_title:
            filtered_data = filtered_data[
                filtered_data['job_title_clean'].str.contains(job_title, case=False, na=False)
            ]
        
        if location:
            filtered_data = filtered_data[
                filtered_data['city'].str.contains(location, case=False, na=False)
            ]
        
        if industry and 'industry' in filtered_data.columns:
            filtered_data = filtered_data[
                filtered_data['industry'].str.contains(industry, case=False, na=False)
            ]
        
        if len(filtered_data) == 0:
            return {
                "prediction": None,
                "message": "Không tìm thấy dữ liệu phù hợp",
                "sample_size": 0
            }
        
        # Calculate salary statistics
        if 'salary_min' in filtered_data.columns and 'salary_max' in filtered_data.columns:
            salary_min_numeric = pd.to_numeric(filtered_data['salary_min'], errors='coerce')
            salary_max_numeric = pd.to_numeric(filtered_data['salary_max'], errors='coerce')
            avg_salary = (salary_min_numeric + salary_max_numeric) / 2
            
            salary_stats = {
                'predicted_min': float(salary_min_numeric.mean()) if not salary_min_numeric.isna().all() else None,
                'predicted_max': float(salary_max_numeric.mean()) if not salary_max_numeric.isna().all() else None,
                'predicted_avg': float(avg_salary.mean()) if not avg_salary.isna().all() else None,
                'median': float(avg_salary.median()) if not avg_salary.isna().all() else None,
                'std': float(avg_salary.std()) if not avg_salary.isna().all() else None
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
                "message": "Dữ liệu lương không có sẵn",
                "sample_size": len(filtered_data)
            }
        
    except Exception as e:
        logger.error(f"Lỗi trong predict_salary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/skills")
async def get_skills_analysis():
    """Lấy phân tích kỹ năng."""
    try:
        if all_data is None or all_data.empty:
            raise HTTPException(status_code=404, detail="Không có dữ liệu")
        
        # Extract skills
        all_skills = []
        for skills_str in all_data['skills'].dropna():
            if isinstance(skills_str, str) and skills_str.strip():
                skills = [skill.strip() for skill in str(skills_str).split(',') if skill.strip()]
                all_skills.extend(skills)
        
        # Count skills
        from collections import Counter
        skills_counter = Counter(all_skills)
        
        # Get top skills
        top_skills = dict(skills_counter.most_common(20))
        
        return {
            "total_unique_skills": len(skills_counter),
            "top_skills": top_skills,
            "sample_skills": dict(skills_counter.most_common(50))
        }
        
    except Exception as e:
        logger.error(f"Lỗi trong get_skills_analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/geographic")
async def get_geographic_analysis():
    """Lấy phân tích địa lý."""
    try:
        if all_data is None or all_data.empty:
            raise HTTPException(status_code=404, detail="Không có dữ liệu")
        
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
                'distribution': city_counts.head(50).to_dict(),
                'top_cities': city_counts.head(20).to_dict()
            }
        
        return {
            "countries": country_stats,
            "cities": city_stats
        }
        
    except Exception as e:
        logger.error(f"Lỗi trong get_geographic_analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
