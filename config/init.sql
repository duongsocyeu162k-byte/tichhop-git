-- Initialize Job Analytics Database
-- This script creates the necessary tables for the job analytics project

-- Note: Database is already created by docker-compose environment variables
-- This script runs after database creation

-- Create raw data tables
CREATE TABLE IF NOT EXISTS raw_glassdoor (
    id SERIAL PRIMARY KEY,
    job_title TEXT,
    salary_estimate TEXT,
    job_description TEXT,
    rating FLOAT,
    company_name TEXT,
    location TEXT,
    headquarters TEXT,
    size TEXT,
    founded INTEGER,
    type_of_ownership TEXT,
    industry TEXT,
    sector TEXT,
    revenue TEXT,
    competitors TEXT,
    easy_apply TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS raw_monster (
    id SERIAL PRIMARY KEY,
    country TEXT,
    country_code TEXT,
    date_added TIMESTAMP,
    has_expired BOOLEAN,
    job_board TEXT,
    job_description TEXT,
    job_title TEXT,
    job_type TEXT,
    location TEXT,
    organization TEXT,
    page_url TEXT,
    salary TEXT,
    sector TEXT,
    uniq_id TEXT UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS raw_naukri (
    id SERIAL PRIMARY KEY,
    company TEXT,
    education TEXT,
    experience TEXT,
    industry TEXT,
    jobdescription TEXT,
    jobid TEXT,
    joblocation_address TEXT,
    jobtitle TEXT,
    numberofpositions INTEGER,
    payrate TEXT,
    postdate TIMESTAMP,
    site_name TEXT,
    skills TEXT,
    uniq_id TEXT UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create processed data tables
CREATE TABLE IF NOT EXISTS processed_jobs (
    id SERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    original_id TEXT,
    job_title TEXT NOT NULL,
    company_name TEXT,
    location TEXT,
    country TEXT,
    salary_min BIGINT,
    salary_max BIGINT,
    salary_currency TEXT DEFAULT 'VND',
    job_type TEXT,
    industry TEXT,
    sector TEXT,
    skills TEXT[],
    education_level TEXT,
    experience_years INTEGER,
    job_description TEXT,
    sentiment_score FLOAT,
    salary_text TEXT,
    experience_text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create analytics tables
CREATE TABLE IF NOT EXISTS salary_analysis (
    id SERIAL PRIMARY KEY,
    job_title TEXT,
    location TEXT,
    industry TEXT,
    avg_salary FLOAT,
    min_salary FLOAT,
    max_salary FLOAT,
    median_salary FLOAT,
    sample_size INTEGER,
    analysis_date DATE DEFAULT CURRENT_DATE
);

CREATE TABLE IF NOT EXISTS skills_analysis (
    id SERIAL PRIMARY KEY,
    skill_name TEXT,
    frequency INTEGER,
    avg_salary FLOAT,
    industries TEXT[],
    analysis_date DATE DEFAULT CURRENT_DATE
);

CREATE TABLE IF NOT EXISTS market_trends (
    id SERIAL PRIMARY KEY,
    metric_name TEXT,
    metric_value FLOAT,
    category TEXT,
    subcategory TEXT,
    period_start DATE,
    period_end DATE,
    analysis_date DATE DEFAULT CURRENT_DATE
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_processed_jobs_job_title ON processed_jobs(job_title);
CREATE INDEX IF NOT EXISTS idx_processed_jobs_location ON processed_jobs(location);
CREATE INDEX IF NOT EXISTS idx_processed_jobs_industry ON processed_jobs(industry);
CREATE INDEX IF NOT EXISTS idx_processed_jobs_salary ON processed_jobs(salary_min, salary_max);
CREATE INDEX IF NOT EXISTS idx_processed_jobs_created_at ON processed_jobs(created_at);

CREATE INDEX IF NOT EXISTS idx_salary_analysis_job_title ON salary_analysis(job_title);
CREATE INDEX IF NOT EXISTS idx_salary_analysis_location ON salary_analysis(location);
CREATE INDEX IF NOT EXISTS idx_skills_analysis_skill_name ON skills_analysis(skill_name);

-- Create views for common queries
CREATE OR REPLACE VIEW v_job_summary AS
SELECT 
    job_title,
    COUNT(*) as job_count,
    AVG(salary_min) as avg_min_salary,
    AVG(salary_max) as avg_max_salary,
    STRING_AGG(DISTINCT industry, ', ') as industries,
    STRING_AGG(DISTINCT location, ', ') as locations
FROM processed_jobs
GROUP BY job_title
ORDER BY job_count DESC;

CREATE OR REPLACE VIEW v_location_analysis AS
SELECT 
    location,
    country,
    COUNT(*) as job_count,
    AVG(salary_min) as avg_min_salary,
    AVG(salary_max) as avg_max_salary,
    STRING_AGG(DISTINCT industry, ', ') as industries
FROM processed_jobs
GROUP BY location, country
ORDER BY job_count DESC;

CREATE OR REPLACE VIEW v_industry_analysis AS
SELECT 
    industry,
    COUNT(*) as job_count,
    AVG(salary_min) as avg_min_salary,
    AVG(salary_max) as avg_max_salary,
    STRING_AGG(DISTINCT job_title, ', ') as job_titles
FROM processed_jobs
WHERE industry IS NOT NULL
GROUP BY industry
ORDER BY job_count DESC;

-- Insert sample data (optional)
-- This can be used for testing purposes
INSERT INTO processed_jobs (source, job_title, company_name, location, country, salary_min, salary_max, industry, skills)
VALUES 
    ('sample', 'Data Scientist', 'Tech Corp', 'San Francisco', 'USA', 80000, 120000, 'Technology', ARRAY['Python', 'Machine Learning', 'SQL']),
    ('sample', 'Data Analyst', 'Analytics Inc', 'New York', 'USA', 60000, 90000, 'Finance', ARRAY['Excel', 'SQL', 'Tableau']),
    ('sample', 'Software Engineer', 'StartupXYZ', 'Bangalore', 'India', 50000, 80000, 'Technology', ARRAY['Java', 'Spring', 'Microservices']);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO admin;
