"""
Advanced Salary Prediction Module
================================

Handles advanced salary prediction using machine learning models
for job market analytics data.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import re
from collections import Counter
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class AdvancedSalaryPredictor:
    """
    A class to perform advanced salary prediction using machine learning.
    """
    
    def __init__(self):
        """Initialize the AdvancedSalaryPredictor."""
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(random_state=42),
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0)
        }
        
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.best_model = None
        self.feature_importance = None
        
        # Salary prediction features
        self.feature_columns = [
            'job_title_clean', 'company_name', 'city', 'state', 'country',
            'industry', 'experience', 'skills', 'job_description'
        ]
        
        # Experience mapping
        self.experience_mapping = {
            'entry': 1,
            'junior': 2,
            'mid': 4,
            'senior': 7,
            'lead': 10,
            'principal': 15,
            'director': 20
        }
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for salary prediction.
        
        Args:
            df: DataFrame with job data
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target variable
        """
        if df.empty:
            return pd.DataFrame(), pd.Series()
        
        logger.info("Preparing features for salary prediction...")
        
        # Create a copy for feature engineering
        features_df = df.copy()
        
        # Calculate target variable (average salary)
        if 'salary_min' in df.columns and 'salary_max' in df.columns:
            features_df['target_salary'] = (df['salary_min'] + df['salary_max']) / 2
        else:
            logger.error("No salary data available for prediction")
            return pd.DataFrame(), pd.Series()
        
        # Remove rows with missing target
        features_df = features_df.dropna(subset=['target_salary'])
        
        if len(features_df) == 0:
            logger.error("No valid salary data found")
            return pd.DataFrame(), pd.Series()
        
        # Feature Engineering
        features_df = self._engineer_features(features_df)
        
        # Select features for modeling
        feature_columns = self._get_feature_columns(features_df)
        X = features_df[feature_columns]
        y = features_df['target_salary']
        
        logger.info(f"Prepared {len(X)} samples with {len(feature_columns)} features")
        return X, y
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for salary prediction."""
        
        # 1. Job Title Features
        if 'job_title_clean' in df.columns:
            df['title_seniority'] = df['job_title_clean'].apply(self._extract_seniority)
            df['title_category'] = df['job_title_clean'].apply(self._categorize_job_title)
            df['title_length'] = df['job_title_clean'].str.len()
        
        # 2. Location Features
        if 'city' in df.columns:
            df['city_tier'] = df['city'].apply(self._get_city_tier)
        
        if 'country' in df.columns:
            df['country_developed'] = df['country'].apply(self._is_developed_country)
        
        # 3. Experience Features
        if 'experience' in df.columns:
            df['experience_bucket'] = df['experience'].apply(self._bucket_experience)
            df['experience_squared'] = df['experience'] ** 2
        
        # 4. Skills Features
        if 'skills' in df.columns:
            df['skills_count'] = df['skills'].apply(self._count_skills)
            df['has_ml_skills'] = df['skills'].apply(self._has_ml_skills)
            df['has_cloud_skills'] = df['skills'].apply(self._has_cloud_skills)
            df['has_leadership_skills'] = df['skills'].apply(self._has_leadership_skills)
        
        # 5. Company Features
        if 'company_name' in df.columns:
            df['company_size_estimate'] = df['company_name'].apply(self._estimate_company_size)
            df['is_faang'] = df['company_name'].apply(self._is_faang_company)
        
        # 6. Industry Features
        if 'industry' in df.columns:
            df['industry_tech'] = df['industry'].apply(self._is_tech_industry)
            df['industry_finance'] = df['industry'].apply(self._is_finance_industry)
        
        # 7. Job Description Features
        if 'job_description' in df.columns:
            df['description_length'] = df['job_description'].str.len()
            df['has_remote'] = df['job_description'].str.contains('remote|work from home', case=False, na=False)
            df['has_benefits'] = df['job_description'].str.contains('benefits|perks|compensation', case=False, na=False)
            df['has_leadership'] = df['job_description'].str.contains('lead|manage|team', case=False, na=False)
        
        return df
    
    def _extract_seniority(self, title: str) -> str:
        """Extract seniority level from job title."""
        if pd.isna(title):
            return 'unknown'
        
        title_lower = str(title).lower()
        
        if any(word in title_lower for word in ['senior', 'sr', 'lead', 'principal', 'director']):
            return 'senior'
        elif any(word in title_lower for word in ['junior', 'jr', 'entry', 'associate']):
            return 'junior'
        elif any(word in title_lower for word in ['mid', 'intermediate']):
            return 'mid'
        else:
            return 'standard'
    
    def _categorize_job_title(self, title: str) -> str:
        """Categorize job title into broad categories."""
        if pd.isna(title):
            return 'other'
        
        title_lower = str(title).lower()
        
        if any(word in title_lower for word in ['data scientist', 'data analyst', 'data engineer']):
            return 'data'
        elif any(word in title_lower for word in ['software engineer', 'developer', 'programmer']):
            return 'software'
        elif any(word in title_lower for word in ['product manager', 'project manager']):
            return 'management'
        elif any(word in title_lower for word in ['designer', 'ui', 'ux']):
            return 'design'
        elif any(word in title_lower for word in ['marketing', 'sales']):
            return 'business'
        else:
            return 'other'
    
    def _get_city_tier(self, city: str) -> int:
        """Get city tier based on cost of living and tech presence."""
        if pd.isna(city):
            return 3
        
        city_lower = str(city).lower()
        
        # Tier 1 cities (high cost of living, major tech hubs)
        tier1_cities = ['san francisco', 'new york', 'seattle', 'boston', 'los angeles', 'washington']
        if any(tier1 in city_lower for tier1 in tier1_cities):
            return 1
        
        # Tier 2 cities (moderate cost of living, growing tech presence)
        tier2_cities = ['austin', 'denver', 'chicago', 'atlanta', 'dallas', 'phoenix', 'miami']
        if any(tier2 in city_lower for tier2 in tier2_cities):
            return 2
        
        # Tier 3 cities (lower cost of living)
        return 3
    
    def _is_developed_country(self, country: str) -> bool:
        """Check if country is developed."""
        if pd.isna(country):
            return False
        
        developed_countries = ['united states', 'canada', 'united kingdom', 'germany', 'france', 
                             'australia', 'japan', 'singapore', 'switzerland', 'netherlands']
        return str(country).lower() in developed_countries
    
    def _bucket_experience(self, exp: float) -> str:
        """Bucket experience into categories."""
        if pd.isna(exp):
            return 'unknown'
        
        if exp < 2:
            return 'entry'
        elif exp < 5:
            return 'junior'
        elif exp < 10:
            return 'mid'
        else:
            return 'senior'
    
    def _count_skills(self, skills: str) -> int:
        """Count number of skills."""
        if pd.isna(skills) or skills == '':
            return 0
        return len([s.strip() for s in str(skills).split(',') if s.strip()])
    
    def _has_ml_skills(self, skills: str) -> bool:
        """Check if has machine learning skills."""
        if pd.isna(skills):
            return False
        
        ml_keywords = ['machine learning', 'ml', 'deep learning', 'ai', 'artificial intelligence', 
                      'tensorflow', 'pytorch', 'scikit-learn', 'neural network']
        skills_lower = str(skills).lower()
        return any(keyword in skills_lower for keyword in ml_keywords)
    
    def _has_cloud_skills(self, skills: str) -> bool:
        """Check if has cloud skills."""
        if pd.isna(skills):
            return False
        
        cloud_keywords = ['aws', 'azure', 'gcp', 'google cloud', 'amazon web services', 'docker', 'kubernetes']
        skills_lower = str(skills).lower()
        return any(keyword in skills_lower for keyword in cloud_keywords)
    
    def _has_leadership_skills(self, skills: str) -> bool:
        """Check if has leadership skills."""
        if pd.isna(skills):
            return False
        
        leadership_keywords = ['leadership', 'management', 'team lead', 'mentoring', 'coaching']
        skills_lower = str(skills).lower()
        return any(keyword in skills_lower for keyword in leadership_keywords)
    
    def _estimate_company_size(self, company: str) -> str:
        """Estimate company size based on name."""
        if pd.isna(company):
            return 'unknown'
        
        company_lower = str(company).lower()
        
        # Large companies (FAANG, Fortune 500)
        large_companies = ['google', 'microsoft', 'amazon', 'apple', 'meta', 'facebook', 'netflix', 
                          'oracle', 'salesforce', 'adobe', 'intel', 'cisco', 'ibm']
        if any(large in company_lower for large in large_companies):
            return 'large'
        
        # Medium companies (well-known but not FAANG)
        medium_companies = ['uber', 'airbnb', 'spotify', 'twitter', 'linkedin', 'paypal', 'stripe']
        if any(medium in company_lower for medium in medium_companies):
            return 'medium'
        
        return 'small'
    
    def _is_faang_company(self, company: str) -> bool:
        """Check if company is FAANG."""
        if pd.isna(company):
            return False
        
        faang_companies = ['google', 'microsoft', 'amazon', 'apple', 'meta', 'facebook', 'netflix']
        return str(company).lower() in faang_companies
    
    def _is_tech_industry(self, industry: str) -> bool:
        """Check if industry is technology."""
        if pd.isna(industry):
            return False
        
        tech_keywords = ['technology', 'software', 'it', 'tech', 'computer', 'internet', 'digital']
        industry_lower = str(industry).lower()
        return any(keyword in industry_lower for keyword in tech_keywords)
    
    def _is_finance_industry(self, industry: str) -> bool:
        """Check if industry is finance."""
        if pd.isna(industry):
            return False
        
        finance_keywords = ['finance', 'banking', 'financial', 'investment', 'insurance']
        industry_lower = str(industry).lower()
        return any(keyword in industry_lower for keyword in finance_keywords)
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns for modeling."""
        # Categorical features
        categorical_features = [
            'title_seniority', 'title_category', 'city_tier', 'country_developed',
            'experience_bucket', 'company_size_estimate', 'is_faang', 'industry_tech', 'industry_finance'
        ]
        
        # Numerical features
        numerical_features = [
            'experience', 'experience_squared', 'skills_count', 'title_length',
            'description_length'
        ]
        
        # Boolean features
        boolean_features = [
            'has_ml_skills', 'has_cloud_skills', 'has_leadership_skills',
            'has_remote', 'has_benefits', 'has_leadership'
        ]
        
        # Select only features that exist in the dataframe
        available_features = []
        for feature_list in [categorical_features, numerical_features, boolean_features]:
            for feature in feature_list:
                if feature in df.columns:
                    available_features.append(feature)
        
        return available_features
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Train multiple models and select the best one.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Dict[str, Any]: Training results and model performance
        """
        if X.empty or len(y) == 0:
            return {'error': 'No data available for training'}
        
        logger.info("Training salary prediction models...")
        
        # Handle categorical variables
        X_processed = self._preprocess_features(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )
        
        # Train and evaluate models
        model_results = {}
        
        for name, model in self.models.items():
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_processed, y, cv=5, scoring='r2')
                
                model_results[name] = {
                    'model': model,
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'predictions': y_pred
                }
                
                logger.info(f"{name}: R² = {r2:.3f}, MAE = ${mae:,.0f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                model_results[name] = {'error': str(e)}
        
        # Select best model based on R² score
        best_model_name = max(
            [name for name, result in model_results.items() if 'error' not in result],
            key=lambda x: model_results[x]['r2']
        )
        
        self.best_model = model_results[best_model_name]['model']
        
        # Get feature importance if available
        if hasattr(self.best_model, 'feature_importances_'):
            feature_names = X_processed.columns
            self.feature_importance = dict(zip(feature_names, self.best_model.feature_importances_))
        
        return {
            'model_results': model_results,
            'best_model': best_model_name,
            'best_model_score': model_results[best_model_name]['r2'],
            'feature_importance': self.feature_importance,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
    
    def _preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Preprocess features for modeling."""
        X_processed = X.copy()
        
        # Handle categorical variables
        categorical_columns = X_processed.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X_processed[col] = self.label_encoders[col].fit_transform(X_processed[col].astype(str))
            else:
                # Handle unseen categories
                X_processed[col] = X_processed[col].astype(str)
                X_processed[col] = X_processed[col].apply(
                    lambda x: x if x in self.label_encoders[col].classes_ else 'unknown'
                )
                X_processed[col] = self.label_encoders[col].transform(X_processed[col])
        
        # Scale numerical features
        numerical_columns = X_processed.select_dtypes(include=[np.number]).columns
        if len(numerical_columns) > 0:
            X_processed[numerical_columns] = self.scaler.fit_transform(X_processed[numerical_columns])
        
        return X_processed
    
    def predict_salary(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict salary for a given job.
        
        Args:
            job_data: Dictionary with job information
            
        Returns:
            Dict[str, Any]: Salary prediction results
        """
        if self.best_model is None:
            return {'error': 'No trained model available'}
        
        try:
            # Convert job data to DataFrame
            job_df = pd.DataFrame([job_data])
            
            # Engineer features
            job_df = self._engineer_features(job_df)
            
            # Get feature columns
            feature_columns = self._get_feature_columns(job_df)
            X = job_df[feature_columns]
            
            # Preprocess features
            X_processed = self._preprocess_features(X)
            
            # Make prediction
            predicted_salary = self.best_model.predict(X_processed)[0]
            
            # Calculate confidence interval (simplified)
            confidence_interval = predicted_salary * 0.15  # ±15% confidence
            
            return {
                'predicted_salary': predicted_salary,
                'confidence_interval': confidence_interval,
                'salary_range': {
                    'min': predicted_salary - confidence_interval,
                    'max': predicted_salary + confidence_interval
                },
                'features_used': feature_columns,
                'model_used': type(self.best_model).__name__
            }
            
        except Exception as e:
            logger.error(f"Error predicting salary: {e}")
            return {'error': str(e)}
    
    def analyze_salary_factors(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze factors that influence salary.
        
        Args:
            df: DataFrame with job data
            
        Returns:
            Dict[str, Any]: Salary factor analysis results
        """
        if df.empty:
            return {'error': 'No data available'}
        
        logger.info("Analyzing salary factors...")
        
        # Prepare features and target
        X, y = self.prepare_features(df)
        
        if X.empty or len(y) == 0:
            return {'error': 'No valid salary data for analysis'}
        
        # Train model to get feature importance
        training_results = self.train_models(X, y)
        
        if 'error' in training_results:
            return training_results
        
        # Analyze salary by different factors
        factor_analysis = {}
        
        # 1. Salary by job title
        if 'job_title_clean' in df.columns:
            title_salary = df.groupby('job_title_clean')['target_salary'].agg(['mean', 'median', 'count']).reset_index()
            title_salary = title_salary[title_salary['count'] >= 5].sort_values('mean', ascending=False)
            factor_analysis['salary_by_title'] = title_salary.head(20).to_dict('records')
        
        # 2. Salary by location
        if 'city' in df.columns:
            city_salary = df.groupby('city')['target_salary'].agg(['mean', 'median', 'count']).reset_index()
            city_salary = city_salary[city_salary['count'] >= 5].sort_values('mean', ascending=False)
            factor_analysis['salary_by_location'] = city_salary.head(20).to_dict('records')
        
        # 3. Salary by experience
        if 'experience' in df.columns:
            exp_salary = df.groupby('experience')['target_salary'].agg(['mean', 'median', 'count']).reset_index()
            exp_salary = exp_salary[exp_salary['count'] >= 5].sort_values('experience')
            factor_analysis['salary_by_experience'] = exp_salary.to_dict('records')
        
        # 4. Salary by skills
        if 'skills' in df.columns:
            skill_salary_analysis = self._analyze_skill_salary_impact(df)
            factor_analysis['salary_by_skills'] = skill_salary_analysis
        
        return {
            'feature_importance': self.feature_importance,
            'model_performance': training_results,
            'factor_analysis': factor_analysis,
            'total_samples': len(df)
        }
    
    def _analyze_skill_salary_impact(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze impact of skills on salary."""
        skill_salary_impact = {}
        
        # Common skills to analyze
        common_skills = ['python', 'java', 'javascript', 'machine learning', 'aws', 'docker', 'kubernetes']
        
        for skill in common_skills:
            # Jobs with this skill
            has_skill = df[df['skills'].str.contains(skill, case=False, na=False)]
            no_skill = df[~df['skills'].str.contains(skill, case=False, na=False)]
            
            if len(has_skill) >= 5 and len(no_skill) >= 5:
                has_skill_avg = has_skill['target_salary'].mean()
                no_skill_avg = no_skill['target_salary'].mean()
                salary_premium = has_skill_avg - no_skill_avg
                
                skill_salary_impact[skill] = {
                    'with_skill_avg': has_skill_avg,
                    'without_skill_avg': no_skill_avg,
                    'salary_premium': salary_premium,
                    'premium_percentage': (salary_premium / no_skill_avg) * 100 if no_skill_avg > 0 else 0,
                    'sample_size': len(has_skill)
                }
        
        return skill_salary_impact
    
    def generate_salary_prediction_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive salary prediction report.
        
        Args:
            df: DataFrame with job data
            
        Returns:
            Dict[str, Any]: Comprehensive salary prediction report
        """
        if df.empty:
            return {'error': 'No data available for salary prediction'}
        
        logger.info("Generating comprehensive salary prediction report...")
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        if X.empty or len(y) == 0:
            return {'error': 'No valid salary data for prediction'}
        
        # Train models
        training_results = self.train_models(X, y)
        
        # Analyze salary factors
        factor_analysis = self.analyze_salary_factors(df)
        
        # Generate insights
        insights = self._generate_salary_insights(training_results, factor_analysis)
        
        report = {
            'summary': {
                'total_jobs_analyzed': len(df),
                'model_performance': training_results.get('best_model_score', 0),
                'best_model': training_results.get('best_model', 'unknown'),
                'analysis_date': datetime.now().isoformat()
            },
            'model_training': training_results,
            'factor_analysis': factor_analysis,
            'insights': insights,
            'feature_importance': self.feature_importance
        }
        
        logger.info("Salary prediction report generated successfully")
        return report
    
    def _generate_salary_insights(self, training_results: Dict, factor_analysis: Dict) -> List[str]:
        """Generate insights from salary prediction analysis."""
        insights = []
        
        # Model performance insights
        best_score = training_results.get('best_model_score', 0)
        if best_score > 0.7:
            insights.append(f"Strong salary prediction model with R² = {best_score:.3f}")
        elif best_score > 0.5:
            insights.append(f"Moderate salary prediction model with R² = {best_score:.3f}")
        else:
            insights.append(f"Weak salary prediction model with R² = {best_score:.3f}")
        
        # Feature importance insights
        if self.feature_importance:
            top_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            insights.append(f"Top salary factors: {', '.join([f[0] for f in top_features])}")
        
        # Salary factor insights
        if 'factor_analysis' in factor_analysis:
            # Top paying job titles
            if 'salary_by_title' in factor_analysis['factor_analysis']:
                top_title = factor_analysis['factor_analysis']['salary_by_title'][0]
                insights.append(f"Highest paying job title: {top_title['job_title_clean']} (${top_title['mean']:,.0f})")
            
            # Top paying locations
            if 'salary_by_location' in factor_analysis['factor_analysis']:
                top_location = factor_analysis['factor_analysis']['salary_by_location'][0]
                insights.append(f"Highest paying location: {top_location['city']} (${top_location['mean']:,.0f})")
            
            # Skill premium insights
            if 'salary_by_skills' in factor_analysis['factor_analysis']:
                skill_impact = factor_analysis['factor_analysis']['salary_by_skills']
                if skill_impact:
                    best_skill = max(skill_impact.items(), key=lambda x: x[1]['salary_premium'])
                    insights.append(f"Highest salary premium skill: {best_skill[0]} (+${best_skill[1]['salary_premium']:,.0f})")
        
        return insights


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Create sample data
    sample_data = pd.DataFrame({
        'job_title_clean': ['Data Scientist', 'Software Engineer', 'Senior Data Scientist'],
        'company_name': ['Google', 'Microsoft', 'Amazon'],
        'city': ['San Francisco', 'Seattle', 'Seattle'],
        'country': ['United States', 'United States', 'United States'],
        'industry': ['Technology', 'Technology', 'Technology'],
        'experience': [3, 5, 7],
        'skills': ['python,machine learning', 'java,spring', 'python,ml,aws'],
        'job_description': ['Data analysis role', 'Software development', 'Senior data role'],
        'salary_min': [80000, 70000, 120000],
        'salary_max': [120000, 100000, 180000]
    })
    
    # Initialize predictor
    predictor = AdvancedSalaryPredictor()
    
    # Generate prediction report
    report = predictor.generate_salary_prediction_report(sample_data)
    
    print("Salary Prediction Report:")
    print(f"Model Performance: {report['summary']['model_performance']:.3f}")
    print(f"Best Model: {report['summary']['best_model']}")
    print(f"Total Jobs: {report['summary']['total_jobs_analyzed']}")
