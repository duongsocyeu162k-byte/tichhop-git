"""
Comprehensive Analytics Module
============================

Integrates all analytics modules for comprehensive job market analysis.
Provides unified interface for all analytics capabilities.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

# Import all analytics modules
from .trend_analyzer import TrendAnalyzer
from .anomaly_detector import AnomalyDetector
from .sentiment_analyzer import SentimentAnalyzer
from .salary_predictor import AdvancedSalaryPredictor
from .fraud_detector import FraudDetector
from .product_potential_analyzer import ProductPotentialAnalyzer

logger = logging.getLogger(__name__)


class ComprehensiveAnalyzer:
    """
    A comprehensive analyzer that integrates all analytics modules.
    """
    
    def __init__(self):
        """Initialize the ComprehensiveAnalyzer."""
        self.trend_analyzer = TrendAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.salary_predictor = AdvancedSalaryPredictor()
        self.fraud_detector = FraudDetector()
        self.product_potential_analyzer = ProductPotentialAnalyzer()
        
        logger.info("ComprehensiveAnalyzer initialized with all analytics modules")
    
    def generate_comprehensive_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive analytics report using all modules.
        
        Args:
            df: DataFrame with job data
            
        Returns:
            Dict[str, Any]: Comprehensive analytics report
        """
        if df.empty:
            return {'error': 'No data available for analysis'}
        
        logger.info("Generating comprehensive analytics report...")
        
        # Initialize report structure
        report = {
            'summary': {
                'total_jobs_analyzed': len(df),
                'analysis_date': datetime.now().isoformat(),
                'modules_used': [
                    'trend_analysis',
                    'anomaly_detection', 
                    'sentiment_analysis',
                    'salary_prediction',
                    'fraud_detection',
                    'product_potential_analysis'
                ]
            }
        }
        
        try:
            # 1. Trend Analysis (Original)
            logger.info("Running trend analysis...")
            trend_report = self.trend_analyzer.generate_trend_report(df)
            report['trend_analysis'] = trend_report
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            report['trend_analysis'] = {'error': str(e)}
        
        try:
            # 2. Anomaly Detection (New)
            logger.info("Running anomaly detection...")
            anomaly_report = self.anomaly_detector.generate_anomaly_report(df)
            report['anomaly_detection'] = anomaly_report
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            report['anomaly_detection'] = {'error': str(e)}
        
        try:
            # 3. Sentiment Analysis (New)
            logger.info("Running sentiment analysis...")
            sentiment_report = self.sentiment_analyzer.generate_sentiment_report(df)
            report['sentiment_analysis'] = sentiment_report
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            report['sentiment_analysis'] = {'error': str(e)}
        
        try:
            # 4. Salary Prediction (Enhanced)
            logger.info("Running salary prediction analysis...")
            salary_report = self.salary_predictor.generate_salary_prediction_report(df)
            report['salary_prediction'] = salary_report
            
        except Exception as e:
            logger.error(f"Error in salary prediction: {e}")
            report['salary_prediction'] = {'error': str(e)}
        
        try:
            # 5. Fraud Detection (New)
            logger.info("Running fraud detection...")
            fraud_report = self.fraud_detector.generate_fraud_detection_report(df)
            report['fraud_detection'] = fraud_report
            
        except Exception as e:
            logger.error(f"Error in fraud detection: {e}")
            report['fraud_detection'] = {'error': str(e)}
        
        try:
            # 6. Product Potential Analysis (New)
            logger.info("Running product potential analysis...")
            product_report = self.product_potential_analyzer.generate_product_potential_report(df)
            report['product_potential_analysis'] = product_report
            
        except Exception as e:
            logger.error(f"Error in product potential analysis: {e}")
            report['product_potential_analysis'] = {'error': str(e)}
        
        # Generate overall insights
        report['overall_insights'] = self._generate_overall_insights(report)
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(report)
        
        logger.info("Comprehensive analytics report generated successfully")
        return report
    
    def _generate_overall_insights(self, report: Dict[str, Any]) -> List[str]:
        """Generate overall insights from all analytics modules."""
        insights = []
        
        # Data quality insights
        total_jobs = report['summary']['total_jobs_analyzed']
        insights.append(f"Analyzed {total_jobs:,} job postings across multiple data sources")
        
        # Anomaly insights
        if 'anomaly_detection' in report and 'error' not in report['anomaly_detection']:
            anomaly_summary = report['anomaly_detection'].get('summary', {})
            total_anomalies = anomaly_summary.get('total_anomalies', 0)
            anomaly_rate = anomaly_summary.get('anomaly_rate', 0)
            
            if anomaly_rate > 10:
                insights.append(f"High anomaly rate detected: {anomaly_rate:.1f}% of data shows unusual patterns")
            elif anomaly_rate > 5:
                insights.append(f"Moderate anomaly rate: {anomaly_rate:.1f}% of data shows unusual patterns")
            else:
                insights.append(f"Low anomaly rate: {anomaly_rate:.1f}% of data shows unusual patterns")
        
        # Sentiment insights
        if 'sentiment_analysis' in report and 'error' not in report['sentiment_analysis']:
            sentiment_summary = report['sentiment_analysis'].get('summary', {})
            positive_pct = sentiment_summary.get('positive_percentage', 0)
            negative_pct = sentiment_summary.get('negative_percentage', 0)
            
            if positive_pct > 60:
                insights.append(f"Positive job market sentiment: {positive_pct:.1f}% of job descriptions are positive")
            elif negative_pct > 40:
                insights.append(f"Concerning job market sentiment: {negative_pct:.1f}% of job descriptions are negative")
            else:
                insights.append(f"Balanced job market sentiment: {positive_pct:.1f}% positive, {negative_pct:.1f}% negative")
        
        # Fraud insights
        if 'fraud_detection' in report and 'error' not in report['fraud_detection']:
            fraud_summary = report['fraud_detection'].get('summary', {})
            fraud_rate = fraud_summary.get('fraud_rate', 0)
            high_risk = fraud_summary.get('high_risk_jobs', 0)
            
            if fraud_rate > 10:
                insights.append(f"High fraud risk detected: {fraud_rate:.1f}% of job postings are suspicious")
            elif fraud_rate > 5:
                insights.append(f"Moderate fraud risk: {fraud_rate:.1f}% of job postings are suspicious")
            else:
                insights.append(f"Low fraud risk: {fraud_rate:.1f}% of job postings are suspicious")
            
            if high_risk > 0:
                insights.append(f"Found {high_risk} high-risk job postings requiring immediate attention")
        
        # Salary insights
        if 'salary_prediction' in report and 'error' not in report['salary_prediction']:
            salary_summary = report['salary_prediction'].get('summary', {})
            model_performance = salary_summary.get('model_performance', 0)
            best_model = salary_summary.get('best_model', 'unknown')
            
            if model_performance > 0.7:
                insights.append(f"Strong salary prediction model ({best_model}) with R² = {model_performance:.3f}")
            elif model_performance > 0.5:
                insights.append(f"Moderate salary prediction model ({best_model}) with R² = {model_performance:.3f}")
            else:
                insights.append(f"Weak salary prediction model ({best_model}) with R² = {model_performance:.3f}")
        
        # Product potential insights
        if 'product_potential_analysis' in report and 'error' not in report['product_potential_analysis']:
            product_summary = report['product_potential_analysis'].get('summary', {})
            market_maturity = product_summary.get('market_maturity', 'unknown')
            growth_potential = product_summary.get('growth_potential', 'unknown')
            
            insights.append(f"Job market maturity: {market_maturity}")
            insights.append(f"Growth potential: {growth_potential}")
        
        return insights
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate recommendations based on all analytics results."""
        recommendations = {
            'data_quality': [],
            'market_insights': [],
            'risk_management': [],
            'career_development': [],
            'business_strategy': []
        }
        
        # Data quality recommendations
        if 'anomaly_detection' in report and 'error' not in report['anomaly_detection']:
            anomaly_summary = report['anomaly_detection'].get('summary', {})
            anomaly_rate = anomaly_summary.get('anomaly_rate', 0)
            
            if anomaly_rate > 10:
                recommendations['data_quality'].append("Implement data validation rules to reduce anomaly rate")
                recommendations['data_quality'].append("Review and clean data sources with high anomaly rates")
            elif anomaly_rate > 5:
                recommendations['data_quality'].append("Monitor data quality and implement basic validation")
        
        # Risk management recommendations
        if 'fraud_detection' in report and 'error' not in report['fraud_detection']:
            fraud_summary = report['fraud_detection'].get('summary', {})
            fraud_rate = fraud_summary.get('fraud_rate', 0)
            high_risk = fraud_summary.get('high_risk_jobs', 0)
            
            if fraud_rate > 10:
                recommendations['risk_management'].append("Implement automated fraud detection system")
                recommendations['risk_management'].append("Review all high-risk job postings manually")
            elif fraud_rate > 5:
                recommendations['risk_management'].append("Implement basic fraud detection filters")
            
            if high_risk > 0:
                recommendations['risk_management'].append(f"Immediately review {high_risk} high-risk job postings")
        
        # Market insights recommendations
        if 'product_potential_analysis' in report and 'error' not in report['product_potential_analysis']:
            product_insights = report['product_potential_analysis'].get('insights', [])
            for insight in product_insights:
                recommendations['market_insights'].append(insight)
        
        # Career development recommendations
        if 'product_potential_analysis' in report and 'error' not in report['product_potential_analysis']:
            career_recs = report['product_potential_analysis'].get('career_recommendations', {})
            
            if 'entry_level' in career_recs:
                recommendations['career_development'].extend(career_recs['entry_level'])
            
            if 'skill_development' in career_recs:
                recommendations['career_development'].extend(career_recs['skill_development'])
            
            if 'location_recommendations' in career_recs:
                recommendations['career_development'].extend(career_recs['location_recommendations'])
        
        # Business strategy recommendations
        if 'sentiment_analysis' in report and 'error' not in report['sentiment_analysis']:
            sentiment_summary = report['sentiment_analysis'].get('summary', {})
            positive_pct = sentiment_summary.get('positive_percentage', 0)
            
            if positive_pct < 50:
                recommendations['business_strategy'].append("Improve job posting descriptions to be more positive and attractive")
                recommendations['business_strategy'].append("Focus on highlighting company benefits and growth opportunities")
        
        if 'salary_prediction' in report and 'error' not in report['salary_prediction']:
            salary_summary = report['salary_prediction'].get('summary', {})
            model_performance = salary_summary.get('model_performance', 0)
            
            if model_performance > 0.7:
                recommendations['business_strategy'].append("Use salary prediction model for competitive compensation analysis")
                recommendations['business_strategy'].append("Implement dynamic salary recommendations based on market data")
        
        return recommendations
    
    def get_analytics_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Get a summary of all analytics results."""
        summary = {
            'total_jobs': report['summary']['total_jobs_analyzed'],
            'analysis_date': report['summary']['analysis_date'],
            'modules_status': {},
            'key_metrics': {},
            'alerts': []
        }
        
        # Check module status
        for module in report['summary']['modules_used']:
            if module in report and 'error' not in report[module]:
                summary['modules_status'][module] = 'success'
            else:
                summary['modules_status'][module] = 'error'
                summary['alerts'].append(f"{module} module failed")
        
        # Extract key metrics
        if 'anomaly_detection' in report and 'error' not in report['anomaly_detection']:
            anomaly_summary = report['anomaly_detection'].get('summary', {})
            summary['key_metrics']['anomaly_rate'] = anomaly_summary.get('anomaly_rate', 0)
            summary['key_metrics']['total_anomalies'] = anomaly_summary.get('total_anomalies', 0)
        
        if 'sentiment_analysis' in report and 'error' not in report['sentiment_analysis']:
            sentiment_summary = report['sentiment_analysis'].get('summary', {})
            summary['key_metrics']['positive_sentiment'] = sentiment_summary.get('positive_percentage', 0)
            summary['key_metrics']['negative_sentiment'] = sentiment_summary.get('negative_percentage', 0)
        
        if 'fraud_detection' in report and 'error' not in report['fraud_detection']:
            fraud_summary = report['fraud_detection'].get('summary', {})
            summary['key_metrics']['fraud_rate'] = fraud_summary.get('fraud_rate', 0)
            summary['key_metrics']['high_risk_jobs'] = fraud_summary.get('high_risk_jobs', 0)
        
        if 'salary_prediction' in report and 'error' not in report['salary_prediction']:
            salary_summary = report['salary_prediction'].get('summary', {})
            summary['key_metrics']['salary_model_performance'] = salary_summary.get('model_performance', 0)
        
        if 'product_potential_analysis' in report and 'error' not in report['product_potential_analysis']:
            product_summary = report['product_potential_analysis'].get('summary', {})
            summary['key_metrics']['market_maturity'] = product_summary.get('market_maturity', 'unknown')
            summary['key_metrics']['growth_potential'] = product_summary.get('growth_potential', 'unknown')
        
        # Add alerts for high-risk situations
        if summary['key_metrics'].get('fraud_rate', 0) > 10:
            summary['alerts'].append("High fraud rate detected - immediate attention required")
        
        if summary['key_metrics'].get('anomaly_rate', 0) > 15:
            summary['alerts'].append("High anomaly rate - data quality issues detected")
        
        if summary['key_metrics'].get('negative_sentiment', 0) > 50:
            summary['alerts'].append("High negative sentiment - market conditions may be challenging")
        
        return summary


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
        'job_description': [
            'Join our amazing team! Great benefits and growth opportunities.',
            'We are looking for a talented engineer to join our innovative team.',
            'Challenging role with tight deadlines and high pressure environment.'
        ],
        'salary_min': [80000, 70000, 120000],
        'salary_max': [120000, 100000, 180000],
        'source': ['glassdoor', 'monster', 'naukri']
    })
    
    # Initialize comprehensive analyzer
    analyzer = ComprehensiveAnalyzer()
    
    # Generate comprehensive report
    report = analyzer.generate_comprehensive_report(sample_data)
    
    # Get summary
    summary = analyzer.get_analytics_summary(report)
    
    print("Comprehensive Analytics Summary:")
    print(f"Total Jobs: {summary['total_jobs']}")
    print(f"Modules Status: {summary['modules_status']}")
    print(f"Key Metrics: {summary['key_metrics']}")
    if summary['alerts']:
        print(f"Alerts: {summary['alerts']}")
