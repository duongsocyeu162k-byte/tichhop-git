"""
Product Potential Analysis Module
===============================

Handles product potential evaluation and prediction for job market analytics.
Analyzes job market demand, skill trends, and career path optimization.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ProductPotentialAnalyzer:
    """
    A class to analyze and predict product potential in the job market.
    """
    
    def __init__(self):
        """Initialize the ProductPotentialAnalyzer."""
        self.demand_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.skill_trend_predictor = GradientBoostingRegressor(random_state=42)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        # Market categories for analysis
        self.market_categories = {
            'data_science': ['data scientist', 'data analyst', 'data engineer', 'ml engineer'],
            'software_development': ['software engineer', 'developer', 'programmer', 'full stack'],
            'product_management': ['product manager', 'product owner', 'scrum master'],
            'design': ['ui designer', 'ux designer', 'graphic designer', 'product designer'],
            'marketing': ['marketing manager', 'digital marketing', 'growth hacker'],
            'sales': ['sales manager', 'account executive', 'business development'],
            'operations': ['operations manager', 'project manager', 'business analyst'],
            'finance': ['financial analyst', 'accountant', 'financial manager'],
            'hr': ['hr manager', 'recruiter', 'talent acquisition'],
            'cybersecurity': ['security engineer', 'cybersecurity analyst', 'penetration tester']
        }
        
        # Skill categories
        self.skill_categories = {
            'programming_languages': ['python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust'],
            'data_science': ['machine learning', 'deep learning', 'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch'],
            'cloud_platforms': ['aws', 'azure', 'gcp', 'google cloud', 'amazon web services'],
            'databases': ['mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'sql'],
            'web_technologies': ['react', 'angular', 'vue', 'node.js', 'django', 'flask', 'spring'],
            'devops': ['docker', 'kubernetes', 'jenkins', 'git', 'ci/cd', 'terraform'],
            'mobile': ['ios', 'android', 'react native', 'flutter', 'swift', 'kotlin'],
            'ai_ml': ['artificial intelligence', 'machine learning', 'neural networks', 'nlp', 'computer vision']
        }
        
        # Market maturity levels
        self.maturity_levels = {
            'emerging': 1,
            'growing': 2,
            'mature': 3,
            'declining': 4
        }
    
    def analyze_job_market_demand(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze job market demand trends.
        
        Args:
            df: DataFrame with job data
            
        Returns:
            Dict[str, Any]: Job market demand analysis results
        """
        if df.empty:
            return {'error': 'No data available'}
        
        logger.info("Analyzing job market demand...")
        
        demand_analysis = {}
        
        # 1. Overall market demand
        overall_demand = self._analyze_overall_demand(df)
        demand_analysis['overall_demand'] = overall_demand
        
        # 2. Demand by job categories
        category_demand = self._analyze_category_demand(df)
        demand_analysis['category_demand'] = category_demand
        
        # 3. Demand by skills
        skill_demand = self._analyze_skill_demand(df)
        demand_analysis['skill_demand'] = skill_demand
        
        # 4. Demand by location
        location_demand = self._analyze_location_demand(df)
        demand_analysis['location_demand'] = location_demand
        
        # 5. Demand by experience level
        experience_demand = self._analyze_experience_demand(df)
        demand_analysis['experience_demand'] = experience_demand
        
        # 6. Market saturation analysis
        saturation_analysis = self._analyze_market_saturation(df)
        demand_analysis['market_saturation'] = saturation_analysis
        
        return demand_analysis
    
    def _analyze_overall_demand(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze overall market demand."""
        total_jobs = len(df)
        
        # Analyze by source (proxy for time)
        source_distribution = df['source'].value_counts().to_dict() if 'source' in df.columns else {}
        
        # Analyze by industry
        industry_distribution = df['industry'].value_counts().to_dict() if 'industry' in df.columns else {}
        
        # Analyze by company size
        company_size_distribution = df['company_size'].value_counts().to_dict() if 'company_size' in df.columns else {}
        
        return {
            'total_jobs': total_jobs,
            'source_distribution': source_distribution,
            'industry_distribution': industry_distribution,
            'company_size_distribution': company_size_distribution,
            'market_size_estimate': total_jobs * 1000  # Rough estimate
        }
    
    def _analyze_category_demand(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze demand by job categories."""
        if 'job_title_clean' not in df.columns:
            return {'error': 'No job title data available'}
        
        category_counts = defaultdict(int)
        category_salaries = defaultdict(list)
        
        for _, row in df.iterrows():
            title = str(row.get('job_title_clean', '')).lower()
            
            # Categorize job title
            category = self._categorize_job_title(title)
            category_counts[category] += 1
            
            # Collect salary data
            if 'salary_min' in df.columns and 'salary_max' in df.columns:
                salary_min = pd.to_numeric(row.get('salary_min'), errors='coerce')
                salary_max = pd.to_numeric(row.get('salary_max'), errors='coerce')
                if pd.notna(salary_min) and pd.notna(salary_max):
                    avg_salary = (salary_min + salary_max) / 2
                    category_salaries[category].append(avg_salary)
        
        # Calculate category statistics
        category_stats = {}
        for category, count in category_counts.items():
            salaries = category_salaries[category]
            category_stats[category] = {
                'job_count': count,
                'percentage': count / len(df) * 100,
                'avg_salary': np.mean(salaries) if salaries else 0,
                'median_salary': np.median(salaries) if salaries else 0,
                'salary_range': [np.min(salaries), np.max(salaries)] if salaries else [0, 0]
            }
        
        # Sort by job count
        sorted_categories = sorted(category_stats.items(), key=lambda x: x[1]['job_count'], reverse=True)
        
        return {
            'category_statistics': dict(sorted_categories),
            'top_categories': [cat for cat, _ in sorted_categories[:10]],
            'emerging_categories': self._identify_emerging_categories(category_stats)
        }
    
    def _categorize_job_title(self, title: str) -> str:
        """Categorize job title into market categories."""
        title_lower = title.lower()
        
        for category, keywords in self.market_categories.items():
            if any(keyword in title_lower for keyword in keywords):
                return category
        
        return 'other'
    
    def _identify_emerging_categories(self, category_stats: Dict) -> List[str]:
        """Identify emerging job categories."""
        # Simple heuristic: categories with moderate job count but high average salary
        emerging = []
        
        for category, stats in category_stats.items():
            if (stats['job_count'] > 50 and  # Not too small
                stats['avg_salary'] > 80000 and  # High salary
                category not in ['software_development', 'data_science']):  # Not already mainstream
                emerging.append(category)
        
        return emerging
    
    def _analyze_skill_demand(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze demand by skills."""
        if 'skills' not in df.columns:
            return {'error': 'No skills data available'}
        
        # Count all skills
        all_skills = []
        skill_salaries = defaultdict(list)
        
        for _, row in df.iterrows():
            skills_str = str(row.get('skills', ''))
            if skills_str and skills_str != 'nan':
                skills = [skill.strip().lower() for skill in skills_str.split(',') if skill.strip()]
                all_skills.extend(skills)
                
                # Collect salary data for each skill
                if 'salary_min' in df.columns and 'salary_max' in df.columns:
                    salary_min = pd.to_numeric(row.get('salary_min'), errors='coerce')
                    salary_max = pd.to_numeric(row.get('salary_max'), errors='coerce')
                    if pd.notna(salary_min) and pd.notna(salary_max):
                        avg_salary = (salary_min + salary_max) / 2
                        for skill in skills:
                            skill_salaries[skill].append(avg_salary)
        
        # Calculate skill statistics
        skill_counts = Counter(all_skills)
        skill_stats = {}
        
        for skill, count in skill_counts.most_common(50):  # Top 50 skills
            salaries = skill_salaries[skill]
            skill_stats[skill] = {
                'demand_count': count,
                'demand_percentage': count / len(df) * 100,
                'avg_salary': np.mean(salaries) if salaries else 0,
                'median_salary': np.median(salaries) if salaries else 0,
                'salary_premium': self._calculate_salary_premium(skill, skill_salaries, df)
            }
        
        # Categorize skills
        skill_categories = self._categorize_skills(skill_stats)
        
        return {
            'top_skills': dict(list(skill_stats.items())[:20]),
            'skill_categories': skill_categories,
            'emerging_skills': self._identify_emerging_skills(skill_stats),
            'declining_skills': self._identify_declining_skills(skill_stats)
        }
    
    def _calculate_salary_premium(self, skill: str, skill_salaries: Dict, df: pd.DataFrame) -> float:
        """Calculate salary premium for a skill."""
        if skill not in skill_salaries or not skill_salaries[skill]:
            return 0
        
        skill_avg = np.mean(skill_salaries[skill])
        
        # Calculate overall average salary
        if 'salary_min' in df.columns and 'salary_max' in df.columns:
            overall_salaries = []
            for _, row in df.iterrows():
                salary_min = pd.to_numeric(row.get('salary_min'), errors='coerce')
                salary_max = pd.to_numeric(row.get('salary_max'), errors='coerce')
                if pd.notna(salary_min) and pd.notna(salary_max):
                    overall_salaries.append((salary_min + salary_max) / 2)
            
            if overall_salaries:
                overall_avg = np.mean(overall_salaries)
                return ((skill_avg - overall_avg) / overall_avg) * 100
        
        return 0
    
    def _categorize_skills(self, skill_stats: Dict) -> Dict[str, List]:
        """Categorize skills into skill categories."""
        categorized_skills = defaultdict(list)
        
        for skill, stats in skill_stats.items():
            for category, keywords in self.skill_categories.items():
                if any(keyword in skill for keyword in keywords):
                    categorized_skills[category].append({
                        'skill': skill,
                        'demand_count': stats['demand_count'],
                        'avg_salary': stats['avg_salary']
                    })
                    break
        
        # Sort skills within each category by demand
        for category in categorized_skills:
            categorized_skills[category].sort(key=lambda x: x['demand_count'], reverse=True)
        
        return dict(categorized_skills)
    
    def _identify_emerging_skills(self, skill_stats: Dict) -> List[str]:
        """Identify emerging skills."""
        emerging = []
        
        for skill, stats in skill_stats.items():
            # Skills with moderate demand but high salary premium
            if (stats['demand_count'] > 20 and  # Not too rare
                stats['salary_premium'] > 20 and  # High salary premium
                any(keyword in skill for keyword in ['ai', 'ml', 'cloud', 'devops', 'kubernetes', 'docker'])):
                emerging.append(skill)
        
        return emerging[:10]
    
    def _identify_declining_skills(self, skill_stats: Dict) -> List[str]:
        """Identify declining skills."""
        declining = []
        
        for skill, stats in skill_stats.items():
            # Skills with low demand and low salary premium
            if (stats['demand_count'] < 50 and  # Low demand
                stats['salary_premium'] < -10 and  # Negative salary premium
                any(keyword in skill for keyword in ['legacy', 'old', 'deprecated'])):
                declining.append(skill)
        
        return declining[:10]
    
    def _analyze_location_demand(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze demand by location."""
        if 'city' not in df.columns:
            return {'error': 'No location data available'}
        
        location_stats = {}
        
        for city, city_df in df.groupby('city'):
            if len(city_df) < 5:  # Skip cities with too few jobs
                continue
            
            # Calculate city statistics
            location_stats[city] = {
                'job_count': len(city_df),
                'percentage': len(city_df) / len(df) * 100,
                'avg_salary': self._calculate_avg_salary(city_df),
                'top_skills': self._get_top_skills_for_location(city_df),
                'top_industries': self._get_top_industries_for_location(city_df)
            }
        
        # Sort by job count
        sorted_locations = sorted(location_stats.items(), key=lambda x: x[1]['job_count'], reverse=True)
        
        return {
            'location_statistics': dict(sorted_locations[:20]),  # Top 20 locations
            'tech_hubs': self._identify_tech_hubs(location_stats),
            'emerging_locations': self._identify_emerging_locations(location_stats)
        }
    
    def _calculate_avg_salary(self, df: pd.DataFrame) -> float:
        """Calculate average salary for a location."""
        if 'salary_min' not in df.columns or 'salary_max' not in df.columns:
            return 0
        
        salaries = []
        for _, row in df.iterrows():
            salary_min = pd.to_numeric(row.get('salary_min'), errors='coerce')
            salary_max = pd.to_numeric(row.get('salary_max'), errors='coerce')
            if pd.notna(salary_min) and pd.notna(salary_max):
                salaries.append((salary_min + salary_max) / 2)
        
        return np.mean(salaries) if salaries else 0
    
    def _get_top_skills_for_location(self, df: pd.DataFrame) -> List[str]:
        """Get top skills for a location."""
        if 'skills' not in df.columns:
            return []
        
        all_skills = []
        for _, row in df.iterrows():
            skills_str = str(row.get('skills', ''))
            if skills_str and skills_str != 'nan':
                skills = [skill.strip().lower() for skill in skills_str.split(',') if skill.strip()]
                all_skills.extend(skills)
        
        skill_counts = Counter(all_skills)
        return [skill for skill, _ in skill_counts.most_common(5)]
    
    def _get_top_industries_for_location(self, df: pd.DataFrame) -> List[str]:
        """Get top industries for a location."""
        if 'industry' not in df.columns:
            return []
        
        industry_counts = df['industry'].value_counts()
        return industry_counts.head(3).index.tolist()
    
    def _identify_tech_hubs(self, location_stats: Dict) -> List[str]:
        """Identify tech hubs based on job concentration and salaries."""
        tech_hubs = []
        
        for city, stats in location_stats.items():
            if (stats['job_count'] > 100 and  # High job concentration
                stats['avg_salary'] > 80000 and  # High salaries
                any(skill in ['python', 'javascript', 'java', 'aws'] for skill in stats['top_skills'])):
                tech_hubs.append(city)
        
        return tech_hubs[:10]
    
    def _identify_emerging_locations(self, location_stats: Dict) -> List[str]:
        """Identify emerging job locations."""
        emerging = []
        
        for city, stats in location_stats.items():
            if (50 <= stats['job_count'] <= 200 and  # Moderate job count
                stats['avg_salary'] > 70000 and  # Good salaries
                any(keyword in city.lower() for keyword in ['austin', 'denver', 'atlanta', 'phoenix'])):
                emerging.append(city)
        
        return emerging[:5]
    
    def _analyze_experience_demand(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze demand by experience level."""
        if 'experience' not in df.columns:
            return {'error': 'No experience data available'}
        
        experience_stats = {}
        
        for exp, exp_df in df.groupby('experience'):
            if pd.notna(exp) and len(exp_df) >= 5:
                experience_stats[exp] = {
                    'job_count': len(exp_df),
                    'percentage': len(exp_df) / len(df) * 100,
                    'avg_salary': self._calculate_avg_salary(exp_df),
                    'top_skills': self._get_top_skills_for_location(exp_df),
                    'top_industries': self._get_top_industries_for_location(exp_df)
                }
        
        # Sort by experience level
        sorted_experience = sorted(experience_stats.items(), key=lambda x: x[0])
        
        return {
            'experience_statistics': dict(sorted_experience),
            'entry_level_demand': experience_stats.get(0, {}).get('job_count', 0),
            'senior_level_demand': sum(stats['job_count'] for exp, stats in experience_stats.items() if exp >= 5),
            'experience_gap_analysis': self._analyze_experience_gaps(experience_stats)
        }
    
    def _analyze_experience_gaps(self, experience_stats: Dict) -> Dict[str, Any]:
        """Analyze experience gaps in the market."""
        gaps = {
            'entry_level_saturation': False,
            'mid_level_shortage': False,
            'senior_level_shortage': False
        }
        
        total_jobs = sum(stats['job_count'] for stats in experience_stats.values())
        
        # Check entry level saturation
        entry_jobs = experience_stats.get(0, {}).get('job_count', 0)
        if entry_jobs / total_jobs > 0.4:  # More than 40% entry level
            gaps['entry_level_saturation'] = True
        
        # Check mid-level shortage
        mid_jobs = sum(stats['job_count'] for exp, stats in experience_stats.items() if 2 <= exp <= 4)
        if mid_jobs / total_jobs < 0.2:  # Less than 20% mid-level
            gaps['mid_level_shortage'] = True
        
        # Check senior-level shortage
        senior_jobs = sum(stats['job_count'] for exp, stats in experience_stats.items() if exp >= 5)
        if senior_jobs / total_jobs < 0.3:  # Less than 30% senior-level
            gaps['senior_level_shortage'] = True
        
        return gaps
    
    def _analyze_market_saturation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market saturation levels."""
        saturation_analysis = {}
        
        # Analyze by job categories
        category_saturation = {}
        for category, keywords in self.market_categories.items():
            category_jobs = df[df['job_title_clean'].str.contains('|'.join(keywords), case=False, na=False)]
            
            if len(category_jobs) > 0:
                # Calculate saturation metrics
                avg_salary = self._calculate_avg_salary(category_jobs)
                job_count = len(category_jobs)
                
                # Simple saturation heuristic
                if job_count > 1000 and avg_salary < 80000:
                    saturation_level = 'high'
                elif job_count > 500 and avg_salary < 90000:
                    saturation_level = 'medium'
                else:
                    saturation_level = 'low'
                
                category_saturation[category] = {
                    'job_count': job_count,
                    'avg_salary': avg_salary,
                    'saturation_level': saturation_level
                }
        
        saturation_analysis['category_saturation'] = category_saturation
        
        # Overall market saturation
        total_jobs = len(df)
        overall_avg_salary = self._calculate_avg_salary(df)
        
        if total_jobs > 10000 and overall_avg_salary < 75000:
            overall_saturation = 'high'
        elif total_jobs > 5000 and overall_avg_salary < 85000:
            overall_saturation = 'medium'
        else:
            overall_saturation = 'low'
        
        saturation_analysis['overall_saturation'] = {
            'level': overall_saturation,
            'total_jobs': total_jobs,
            'avg_salary': overall_avg_salary
        }
        
        return saturation_analysis
    
    def predict_market_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict future market trends.
        
        Args:
            df: DataFrame with job data
            
        Returns:
            Dict[str, Any]: Market trend predictions
        """
        if df.empty:
            return {'error': 'No data available'}
        
        logger.info("Predicting market trends...")
        
        # Prepare data for prediction
        trend_data = self._prepare_trend_data(df)
        
        # Predict demand trends
        demand_predictions = self._predict_demand_trends(trend_data)
        
        # Predict skill trends
        skill_predictions = self._predict_skill_trends(trend_data)
        
        # Predict salary trends
        salary_predictions = self._predict_salary_trends(trend_data)
        
        # Predict location trends
        location_predictions = self._predict_location_trends(trend_data)
        
        return {
            'demand_predictions': demand_predictions,
            'skill_predictions': skill_predictions,
            'salary_predictions': salary_predictions,
            'location_predictions': location_predictions,
            'prediction_confidence': self._calculate_prediction_confidence(trend_data)
        }
    
    def _prepare_trend_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for trend prediction."""
        trend_data = {
            'job_counts': {},
            'skill_counts': {},
            'salary_data': {},
            'location_data': {}
        }
        
        # Job counts by category
        for category, keywords in self.market_categories.items():
            category_jobs = df[df['job_title_clean'].str.contains('|'.join(keywords), case=False, na=False)]
            trend_data['job_counts'][category] = len(category_jobs)
        
        # Skill counts
        if 'skills' in df.columns:
            all_skills = []
            for _, row in df.iterrows():
                skills_str = str(row.get('skills', ''))
                if skills_str and skills_str != 'nan':
                    skills = [skill.strip().lower() for skill in skills_str.split(',') if skill.strip()]
                    all_skills.extend(skills)
            
            skill_counts = Counter(all_skills)
            trend_data['skill_counts'] = dict(skill_counts.most_common(50))
        
        # Salary data
        trend_data['salary_data'] = {
            'avg_salary': self._calculate_avg_salary(df),
            'salary_distribution': self._get_salary_distribution(df)
        }
        
        # Location data
        if 'city' in df.columns:
            location_counts = df['city'].value_counts()
            trend_data['location_data'] = dict(location_counts.head(20))
        
        return trend_data
    
    def _get_salary_distribution(self, df: pd.DataFrame) -> Dict[str, float]:
        """Get salary distribution statistics."""
        if 'salary_min' not in df.columns or 'salary_max' not in df.columns:
            return {}
        
        salaries = []
        for _, row in df.iterrows():
            salary_min = pd.to_numeric(row.get('salary_min'), errors='coerce')
            salary_max = pd.to_numeric(row.get('salary_max'), errors='coerce')
            if pd.notna(salary_min) and pd.notna(salary_max):
                salaries.append((salary_min + salary_max) / 2)
        
        if not salaries:
            return {}
        
        return {
            'min': np.min(salaries),
            'max': np.max(salaries),
            'mean': np.mean(salaries),
            'median': np.median(salaries),
            'std': np.std(salaries),
            'q25': np.percentile(salaries, 25),
            'q75': np.percentile(salaries, 75)
        }
    
    def _predict_demand_trends(self, trend_data: Dict) -> Dict[str, Any]:
        """Predict demand trends for job categories."""
        # Simple trend prediction based on current demand
        job_counts = trend_data['job_counts']
        total_jobs = sum(job_counts.values())
        
        predictions = {}
        for category, count in job_counts.items():
            percentage = count / total_jobs * 100 if total_jobs > 0 else 0
            
            # Simple growth prediction based on category
            if category in ['data_science', 'ai_ml', 'cybersecurity']:
                growth_rate = 0.15  # 15% growth
            elif category in ['software_development', 'cloud_platforms']:
                growth_rate = 0.10  # 10% growth
            else:
                growth_rate = 0.05  # 5% growth
            
            predictions[category] = {
                'current_demand': count,
                'current_percentage': percentage,
                'predicted_growth_rate': growth_rate,
                'predicted_demand_1year': int(count * (1 + growth_rate)),
                'predicted_demand_2year': int(count * (1 + growth_rate) ** 2)
            }
        
        return predictions
    
    def _predict_skill_trends(self, trend_data: Dict) -> Dict[str, Any]:
        """Predict skill trends."""
        skill_counts = trend_data['skill_counts']
        
        predictions = {}
        for skill, count in skill_counts.items():
            # Predict growth based on skill category
            if any(keyword in skill for keyword in ['ai', 'ml', 'cloud', 'devops', 'kubernetes']):
                growth_rate = 0.20  # 20% growth for emerging skills
            elif any(keyword in skill for keyword in ['python', 'javascript', 'aws']):
                growth_rate = 0.10  # 10% growth for popular skills
            else:
                growth_rate = 0.05  # 5% growth for other skills
            
            predictions[skill] = {
                'current_demand': count,
                'predicted_growth_rate': growth_rate,
                'predicted_demand_1year': int(count * (1 + growth_rate)),
                'trend_direction': 'growing' if growth_rate > 0.1 else 'stable'
            }
        
        return predictions
    
    def _predict_salary_trends(self, trend_data: Dict) -> Dict[str, Any]:
        """Predict salary trends."""
        salary_data = trend_data['salary_data']
        
        # Simple salary trend prediction
        current_avg = salary_data.get('avg_salary', 0)
        
        # Predict 3% annual salary growth
        growth_rate = 0.03
        
        return {
            'current_avg_salary': current_avg,
            'predicted_growth_rate': growth_rate,
            'predicted_avg_salary_1year': current_avg * (1 + growth_rate),
            'predicted_avg_salary_2year': current_avg * (1 + growth_rate) ** 2,
            'salary_trend': 'increasing'
        }
    
    def _predict_location_trends(self, trend_data: Dict) -> Dict[str, Any]:
        """Predict location trends."""
        location_data = trend_data['location_data']
        
        predictions = {}
        for location, count in location_data.items():
            # Predict growth based on location type
            if any(keyword in location.lower() for keyword in ['san francisco', 'seattle', 'new york']):
                growth_rate = 0.05  # 5% growth for established tech hubs
            elif any(keyword in location.lower() for keyword in ['austin', 'denver', 'atlanta']):
                growth_rate = 0.10  # 10% growth for emerging tech cities
            else:
                growth_rate = 0.03  # 3% growth for other locations
            
            predictions[location] = {
                'current_job_count': count,
                'predicted_growth_rate': growth_rate,
                'predicted_job_count_1year': int(count * (1 + growth_rate)),
                'trend_direction': 'growing' if growth_rate > 0.05 else 'stable'
            }
        
        return predictions
    
    def _calculate_prediction_confidence(self, trend_data: Dict) -> Dict[str, float]:
        """Calculate confidence in predictions."""
        # Simple confidence calculation based on data availability
        confidence_factors = {
            'data_volume': min(len(trend_data['job_counts']), 1.0),
            'skill_coverage': min(len(trend_data['skill_counts']) / 50, 1.0),
            'location_coverage': min(len(trend_data['location_data']) / 20, 1.0)
        }
        
        overall_confidence = np.mean(list(confidence_factors.values()))
        
        return {
            'overall_confidence': overall_confidence,
            'confidence_factors': confidence_factors,
            'confidence_level': 'high' if overall_confidence > 0.7 else 'medium' if overall_confidence > 0.4 else 'low'
        }
    
    def generate_product_potential_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive product potential analysis report.
        
        Args:
            df: DataFrame with job data
            
        Returns:
            Dict[str, Any]: Comprehensive product potential report
        """
        if df.empty:
            return {'error': 'No data available for product potential analysis'}
        
        logger.info("Generating comprehensive product potential analysis report...")
        
        # Analyze current market demand
        demand_analysis = self.analyze_job_market_demand(df)
        
        # Predict future trends
        trend_predictions = self.predict_market_trends(df)
        
        # Generate insights and recommendations
        insights = self._generate_product_insights(demand_analysis, trend_predictions)
        
        # Generate career path recommendations
        career_recommendations = self._generate_career_recommendations(demand_analysis, trend_predictions)
        
        report = {
            'summary': {
                'total_jobs_analyzed': len(df),
                'analysis_date': datetime.now().isoformat(),
                'market_maturity': self._assess_market_maturity(demand_analysis),
                'growth_potential': self._assess_growth_potential(trend_predictions)
            },
            'current_market_analysis': demand_analysis,
            'future_trend_predictions': trend_predictions,
            'insights': insights,
            'career_recommendations': career_recommendations
        }
        
        logger.info("Product potential analysis report generated successfully")
        return report
    
    def _assess_market_maturity(self, demand_analysis: Dict) -> str:
        """Assess overall market maturity."""
        overall_demand = demand_analysis.get('overall_demand', {})
        total_jobs = overall_demand.get('total_jobs', 0)
        avg_salary = overall_demand.get('avg_salary', 0)
        
        if total_jobs > 10000 and avg_salary > 80000:
            return 'mature'
        elif total_jobs > 5000 and avg_salary > 70000:
            return 'growing'
        elif total_jobs > 1000:
            return 'emerging'
        else:
            return 'nascent'
    
    def _assess_growth_potential(self, trend_predictions: Dict) -> str:
        """Assess growth potential based on predictions."""
        demand_predictions = trend_predictions.get('demand_predictions', {})
        
        if not demand_predictions:
            return 'unknown'
        
        avg_growth_rate = np.mean([pred['predicted_growth_rate'] for pred in demand_predictions.values()])
        
        if avg_growth_rate > 0.15:
            return 'high'
        elif avg_growth_rate > 0.10:
            return 'medium'
        else:
            return 'low'
    
    def _generate_product_insights(self, demand_analysis: Dict, trend_predictions: Dict) -> List[str]:
        """Generate insights from product potential analysis."""
        insights = []
        
        # Market demand insights
        category_demand = demand_analysis.get('category_demand', {})
        if 'top_categories' in category_demand:
            top_category = category_demand['top_categories'][0] if category_demand['top_categories'] else 'unknown'
            insights.append(f"Highest demand job category: {top_category}")
        
        # Emerging categories insights
        if 'emerging_categories' in category_demand:
            emerging = category_demand['emerging_categories']
            if emerging:
                insights.append(f"Emerging job categories: {', '.join(emerging)}")
        
        # Skill insights
        skill_demand = demand_analysis.get('skill_demand', {})
        if 'emerging_skills' in skill_demand:
            emerging_skills = skill_demand['emerging_skills']
            if emerging_skills:
                insights.append(f"Emerging skills to watch: {', '.join(emerging_skills[:3])}")
        
        # Location insights
        location_demand = demand_analysis.get('location_demand', {})
        if 'tech_hubs' in location_demand:
            tech_hubs = location_demand['tech_hubs']
            if tech_hubs:
                insights.append(f"Major tech hubs: {', '.join(tech_hubs[:3])}")
        
        # Trend insights
        demand_predictions = trend_predictions.get('demand_predictions', {})
        if demand_predictions:
            fastest_growing = max(demand_predictions.items(), key=lambda x: x[1]['predicted_growth_rate'])
            insights.append(f"Fastest growing category: {fastest_growing[0]} ({fastest_growing[1]['predicted_growth_rate']*100:.1f}% growth)")
        
        return insights
    
    def _generate_career_recommendations(self, demand_analysis: Dict, trend_predictions: Dict) -> Dict[str, Any]:
        """Generate career path recommendations."""
        recommendations = {
            'entry_level': [],
            'mid_level': [],
            'senior_level': [],
            'skill_development': [],
            'location_recommendations': []
        }
        
        # Entry level recommendations
        category_demand = demand_analysis.get('category_demand', {})
        if 'top_categories' in category_demand:
            top_categories = category_demand['top_categories'][:3]
            recommendations['entry_level'] = [
                f"Consider starting in {category} - high demand with good entry opportunities"
                for category in top_categories
            ]
        
        # Skill development recommendations
        skill_demand = demand_analysis.get('skill_demand', {})
        if 'emerging_skills' in skill_demand:
            emerging_skills = skill_demand['emerging_skills'][:5]
            recommendations['skill_development'] = [
                f"Develop skills in {skill} - emerging high-demand skill"
                for skill in emerging_skills
            ]
        
        # Location recommendations
        location_demand = demand_analysis.get('location_demand', {})
        if 'tech_hubs' in location_demand:
            tech_hubs = location_demand['tech_hubs'][:3]
            recommendations['location_recommendations'] = [
                f"Consider {location} - major tech hub with high job concentration"
                for location in tech_hubs
            ]
        
        return recommendations


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Create sample data
    sample_data = pd.DataFrame({
        'job_title_clean': ['Data Scientist', 'Software Engineer', 'Product Manager'],
        'company_name': ['Google', 'Microsoft', 'Amazon'],
        'city': ['San Francisco', 'Seattle', 'Seattle'],
        'industry': ['Technology', 'Technology', 'Technology'],
        'experience': [3, 5, 7],
        'skills': ['python,machine learning', 'java,spring', 'product management,agile'],
        'salary_min': [80000, 70000, 120000],
        'salary_max': [120000, 100000, 180000],
        'source': ['glassdoor', 'monster', 'naukri']
    })
    
    # Initialize analyzer
    analyzer = ProductPotentialAnalyzer()
    
    # Generate product potential report
    report = analyzer.generate_product_potential_report(sample_data)
    
    print("Product Potential Analysis Report:")
    print(f"Market Maturity: {report['summary']['market_maturity']}")
    print(f"Growth Potential: {report['summary']['growth_potential']}")
    print(f"Total Jobs: {report['summary']['total_jobs_analyzed']}")
