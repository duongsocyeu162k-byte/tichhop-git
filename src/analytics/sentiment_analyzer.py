"""
Sentiment Analysis Module
========================

Handles sentiment analysis for job market analytics data.
Analyzes sentiment in job descriptions, company reviews, and market trends.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import re
from collections import Counter
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    A class to perform sentiment analysis on job market data.
    """
    
    def __init__(self):
        """Initialize the SentimentAnalyzer."""
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        
        # Job-specific sentiment keywords
        self.positive_job_keywords = [
            'excellent', 'amazing', 'great', 'fantastic', 'wonderful', 'outstanding',
            'innovative', 'cutting-edge', 'leading', 'premier', 'top-tier', 'elite',
            'competitive', 'attractive', 'comprehensive', 'generous', 'flexible',
            'collaborative', 'dynamic', 'exciting', 'challenging', 'rewarding',
            'growth', 'opportunity', 'advancement', 'development', 'learning',
            'work-life balance', 'remote', 'hybrid', 'benefits', 'perks'
        ]
        
        self.negative_job_keywords = [
            'stressful', 'demanding', 'challenging', 'difficult', 'complex',
            'tight deadlines', 'overtime', 'weekend work', 'on-call',
            'limited', 'restricted', 'mandatory', 'required', 'must',
            'high pressure', 'fast-paced', 'intense', 'competitive',
            'unpaid', 'volunteer', 'internship', 'contract', 'temporary'
        ]
        
        # Company culture keywords
        self.culture_keywords = {
            'positive': [
                'inclusive', 'diverse', 'collaborative', 'supportive', 'friendly',
                'team-oriented', 'open', 'transparent', 'innovative', 'creative',
                'flexible', 'autonomous', 'empowering', 'mentoring', 'learning'
            ],
            'negative': [
                'hierarchical', 'bureaucratic', 'rigid', 'strict', 'formal',
                'competitive', 'aggressive', 'demanding', 'stressful', 'toxic',
                'micromanagement', 'politics', 'unclear', 'chaotic', 'disorganized'
            ]
        }
    
    def analyze_job_description_sentiment(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze sentiment in job descriptions.
        
        Args:
            df: DataFrame with job description data
            
        Returns:
            Dict[str, Any]: Job description sentiment analysis results
        """
        if df.empty or 'job_description' not in df.columns:
            return {'error': 'No job description data available'}
        
        logger.info("Analyzing job description sentiment...")
        
        # Clean and analyze job descriptions
        descriptions = df['job_description'].dropna()
        if len(descriptions) == 0:
            return {'error': 'No valid job descriptions found'}
        
        sentiment_results = []
        
        for idx, description in descriptions.items():
            if pd.notna(description) and str(description).strip():
                sentiment_scores = self._analyze_text_sentiment(str(description))
                
                # Add job-specific analysis
                job_sentiment = self._analyze_job_specific_sentiment(str(description))
                
                sentiment_results.append({
                    'index': idx,
                    'text_length': len(str(description)),
                    'sentiment_scores': sentiment_scores,
                    'job_sentiment': job_sentiment,
                    'overall_sentiment': self._get_overall_sentiment(sentiment_scores, job_sentiment)
                })
        
        # Aggregate results
        aggregated_results = self._aggregate_sentiment_results(sentiment_results)
        
        # Analyze sentiment by job title
        title_sentiment = self._analyze_sentiment_by_title(df, sentiment_results)
        
        # Analyze sentiment by company
        company_sentiment = self._analyze_sentiment_by_company(df, sentiment_results)
        
        return {
            'overall_sentiment': aggregated_results,
            'title_sentiment': title_sentiment,
            'company_sentiment': company_sentiment,
            'individual_results': sentiment_results[:100]  # Limit to first 100 for performance
        }
    
    def _analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using multiple methods."""
        # VADER sentiment analysis
        vader_scores = self.sia.polarity_scores(text)
        
        # TextBlob sentiment analysis
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        # Custom keyword-based analysis
        keyword_scores = self._analyze_keyword_sentiment(text)
        
        return {
            'vader_compound': vader_scores['compound'],
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'vader_neutral': vader_scores['neu'],
            'textblob_polarity': textblob_polarity,
            'textblob_subjectivity': textblob_subjectivity,
            'keyword_positive': keyword_scores['positive'],
            'keyword_negative': keyword_scores['negative'],
            'keyword_ratio': keyword_scores['ratio']
        }
    
    def _analyze_keyword_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment based on job-specific keywords."""
        text_lower = text.lower()
        
        positive_count = sum(1 for keyword in self.positive_job_keywords if keyword in text_lower)
        negative_count = sum(1 for keyword in self.negative_job_keywords if keyword in text_lower)
        
        total_keywords = positive_count + negative_count
        ratio = positive_count / total_keywords if total_keywords > 0 else 0.5
        
        return {
            'positive': positive_count,
            'negative': negative_count,
            'ratio': ratio
        }
    
    def _analyze_job_specific_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze job-specific sentiment aspects."""
        text_lower = text.lower()
        
        # Analyze culture keywords
        culture_sentiment = {
            'positive_culture': sum(1 for keyword in self.culture_keywords['positive'] if keyword in text_lower),
            'negative_culture': sum(1 for keyword in self.culture_keywords['negative'] if keyword in text_lower)
        }
        
        # Analyze work-life balance indicators
        work_life_indicators = {
            'flexible_work': any(keyword in text_lower for keyword in ['remote', 'hybrid', 'flexible', 'work from home']),
            'benefits_mentioned': any(keyword in text_lower for keyword in ['benefits', 'perks', 'compensation', 'package']),
            'growth_opportunities': any(keyword in text_lower for keyword in ['growth', 'development', 'advancement', 'career']),
            'team_collaboration': any(keyword in text_lower for keyword in ['team', 'collaborate', 'together', 'collective'])
        }
        
        # Analyze requirements sentiment
        requirements_sentiment = {
            'must_have_count': len(re.findall(r'\bmust\b|\brequired\b|\bmandatory\b', text_lower)),
            'preferred_count': len(re.findall(r'\bpreferred\b|\bnice to have\b|\bplus\b', text_lower)),
            'experience_mentions': len(re.findall(r'\byears?\b|\bexperience\b|\bexp\b', text_lower))
        }
        
        return {
            'culture_sentiment': culture_sentiment,
            'work_life_balance': work_life_indicators,
            'requirements_sentiment': requirements_sentiment
        }
    
    def _get_overall_sentiment(self, sentiment_scores: Dict[str, float], job_sentiment: Dict[str, Any]) -> str:
        """Determine overall sentiment category."""
        # Combine different sentiment indicators
        vader_compound = sentiment_scores['vader_compound']
        textblob_polarity = sentiment_scores['textblob_polarity']
        keyword_ratio = sentiment_scores['keyword_ratio']
        
        # Weighted average
        overall_score = (vader_compound * 0.4 + textblob_polarity * 0.3 + (keyword_ratio - 0.5) * 2 * 0.3)
        
        # Adjust based on job-specific factors
        culture_positive = job_sentiment['culture_sentiment']['positive_culture']
        culture_negative = job_sentiment['culture_sentiment']['negative_culture']
        
        if culture_positive > culture_negative:
            overall_score += 0.1
        elif culture_negative > culture_positive:
            overall_score -= 0.1
        
        # Categorize sentiment
        if overall_score >= 0.1:
            return 'positive'
        elif overall_score <= -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def _aggregate_sentiment_results(self, sentiment_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate sentiment results across all job descriptions."""
        if not sentiment_results:
            return {'error': 'No sentiment results to aggregate'}
        
        # Count sentiment categories
        sentiment_counts = Counter(result['overall_sentiment'] for result in sentiment_results)
        
        # Calculate average scores
        avg_scores = {}
        for key in ['vader_compound', 'vader_positive', 'vader_negative', 'vader_neutral', 
                   'textblob_polarity', 'textblob_subjectivity', 'keyword_ratio']:
            scores = [result['sentiment_scores'][key] for result in sentiment_results]
            avg_scores[key] = np.mean(scores) if scores else 0
        
        # Analyze work-life balance indicators
        work_life_stats = {
            'flexible_work_percentage': sum(1 for r in sentiment_results 
                                          if r['job_sentiment']['work_life_balance']['flexible_work']) / len(sentiment_results) * 100,
            'benefits_mentioned_percentage': sum(1 for r in sentiment_results 
                                               if r['job_sentiment']['work_life_balance']['benefits_mentioned']) / len(sentiment_results) * 100,
            'growth_opportunities_percentage': sum(1 for r in sentiment_results 
                                                 if r['job_sentiment']['work_life_balance']['growth_opportunities']) / len(sentiment_results) * 100
        }
        
        return {
            'total_analyzed': len(sentiment_results),
            'sentiment_distribution': dict(sentiment_counts),
            'average_scores': avg_scores,
            'work_life_balance_stats': work_life_stats,
            'positive_percentage': sentiment_counts['positive'] / len(sentiment_results) * 100,
            'negative_percentage': sentiment_counts['negative'] / len(sentiment_results) * 100,
            'neutral_percentage': sentiment_counts['neutral'] / len(sentiment_results) * 100
        }
    
    def _analyze_sentiment_by_title(self, df: pd.DataFrame, sentiment_results: List[Dict]) -> Dict[str, Any]:
        """Analyze sentiment by job title."""
        if 'job_title_clean' not in df.columns:
            return {'error': 'No job title data available'}
        
        title_sentiment = {}
        
        # Create mapping from index to sentiment results
        sentiment_map = {result['index']: result for result in sentiment_results}
        
        # Group by job title
        for title, title_df in df.groupby('job_title_clean'):
            if len(title_df) < 3:  # Skip titles with too few samples
                continue
            
            title_sentiments = []
            for idx in title_df.index:
                if idx in sentiment_map:
                    title_sentiments.append(sentiment_map[idx]['overall_sentiment'])
            
            if title_sentiments:
                sentiment_counts = Counter(title_sentiments)
                total = len(title_sentiments)
                
                title_sentiment[title] = {
                    'total_jobs': total,
                    'sentiment_distribution': dict(sentiment_counts),
                    'positive_percentage': sentiment_counts['positive'] / total * 100,
                    'negative_percentage': sentiment_counts['negative'] / total * 100,
                    'neutral_percentage': sentiment_counts['neutral'] / total * 100
                }
        
        return title_sentiment
    
    def _analyze_sentiment_by_company(self, df: pd.DataFrame, sentiment_results: List[Dict]) -> Dict[str, Any]:
        """Analyze sentiment by company."""
        if 'company_name' not in df.columns:
            return {'error': 'No company data available'}
        
        company_sentiment = {}
        
        # Create mapping from index to sentiment results
        sentiment_map = {result['index']: result for result in sentiment_results}
        
        # Group by company
        for company, company_df in df.groupby('company_name'):
            if len(company_df) < 3:  # Skip companies with too few samples
                continue
            
            company_sentiments = []
            for idx in company_df.index:
                if idx in sentiment_map:
                    company_sentiments.append(sentiment_map[idx]['overall_sentiment'])
            
            if company_sentiments:
                sentiment_counts = Counter(company_sentiments)
                total = len(company_sentiments)
                
                company_sentiment[company] = {
                    'total_jobs': total,
                    'sentiment_distribution': dict(sentiment_counts),
                    'positive_percentage': sentiment_counts['positive'] / total * 100,
                    'negative_percentage': sentiment_counts['negative'] / total * 100,
                    'neutral_percentage': sentiment_counts['neutral'] / total * 100
                }
        
        return company_sentiment
    
    def analyze_market_sentiment_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market sentiment trends over time and by source.
        
        Args:
            df: DataFrame with job data
            
        Returns:
            Dict[str, Any]: Market sentiment trends analysis
        """
        if df.empty:
            return {'error': 'No data available'}
        
        logger.info("Analyzing market sentiment trends...")
        
        # Analyze sentiment by source
        source_sentiment = {}
        if 'source' in df.columns:
            for source, source_df in df.groupby('source'):
                source_sentiment[source] = self.analyze_job_description_sentiment(source_df)
        
        # Analyze sentiment by industry
        industry_sentiment = {}
        if 'industry' in df.columns:
            for industry, industry_df in df.groupby('industry'):
                if len(industry_df) >= 5:  # At least 5 jobs
                    industry_sentiment[industry] = self.analyze_job_description_sentiment(industry_df)
        
        # Analyze sentiment by location
        location_sentiment = {}
        if 'city' in df.columns:
            for city, city_df in df.groupby('city'):
                if len(city_df) >= 5:  # At least 5 jobs
                    location_sentiment[city] = self.analyze_job_description_sentiment(city_df)
        
        return {
            'source_sentiment': source_sentiment,
            'industry_sentiment': industry_sentiment,
            'location_sentiment': location_sentiment,
            'analysis_date': datetime.now().isoformat()
        }
    
    def generate_sentiment_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive sentiment analysis report.
        
        Args:
            df: DataFrame with job data
            
        Returns:
            Dict[str, Any]: Comprehensive sentiment report
        """
        if df.empty:
            return {'error': 'No data available for sentiment analysis'}
        
        logger.info("Generating comprehensive sentiment analysis report...")
        
        # Main sentiment analysis
        job_sentiment = self.analyze_job_description_sentiment(df)
        
        # Market sentiment trends
        market_trends = self.analyze_market_sentiment_trends(df)
        
        # Generate insights
        insights = self._generate_sentiment_insights(job_sentiment, market_trends)
        
        report = {
            'summary': {
                'total_jobs_analyzed': job_sentiment.get('overall_sentiment', {}).get('total_analyzed', 0),
                'analysis_date': datetime.now().isoformat(),
                'positive_percentage': job_sentiment.get('overall_sentiment', {}).get('positive_percentage', 0),
                'negative_percentage': job_sentiment.get('overall_sentiment', {}).get('negative_percentage', 0),
                'neutral_percentage': job_sentiment.get('overall_sentiment', {}).get('neutral_percentage', 0)
            },
            'job_description_sentiment': job_sentiment,
            'market_sentiment_trends': market_trends,
            'insights': insights
        }
        
        logger.info("Sentiment analysis report generated successfully")
        return report
    
    def _generate_sentiment_insights(self, job_sentiment: Dict[str, Any], market_trends: Dict[str, Any]) -> List[str]:
        """Generate insights from sentiment analysis."""
        insights = []
        
        # Overall sentiment insights
        overall = job_sentiment.get('overall_sentiment', {})
        positive_pct = overall.get('positive_percentage', 0)
        negative_pct = overall.get('negative_percentage', 0)
        
        if positive_pct > 60:
            insights.append(f"Job market shows positive sentiment with {positive_pct:.1f}% of job descriptions being positive")
        elif negative_pct > 40:
            insights.append(f"Job market shows concerning sentiment with {negative_pct:.1f}% of job descriptions being negative")
        else:
            insights.append(f"Job market shows balanced sentiment with {positive_pct:.1f}% positive and {negative_pct:.1f}% negative")
        
        # Work-life balance insights
        work_life = overall.get('work_life_balance_stats', {})
        flexible_pct = work_life.get('flexible_work_percentage', 0)
        benefits_pct = work_life.get('benefits_mentioned_percentage', 0)
        
        if flexible_pct > 30:
            insights.append(f"Strong work-life balance focus with {flexible_pct:.1f}% of jobs offering flexible work arrangements")
        
        if benefits_pct > 50:
            insights.append(f"Companies are emphasizing benefits with {benefits_pct:.1f}% of job descriptions mentioning benefits")
        
        # Source sentiment insights
        source_sentiment = market_trends.get('source_sentiment', {})
        if source_sentiment:
            best_source = max(source_sentiment.keys(), 
                            key=lambda x: source_sentiment[x].get('overall_sentiment', {}).get('positive_percentage', 0))
            best_pct = source_sentiment[best_source].get('overall_sentiment', {}).get('positive_percentage', 0)
            insights.append(f"Best sentiment found in {best_source} job postings with {best_pct:.1f}% positive sentiment")
        
        return insights


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Create sample data
    sample_data = pd.DataFrame({
        'job_title_clean': ['Data Scientist', 'Software Engineer', 'Product Manager'],
        'company_name': ['Google', 'Microsoft', 'Amazon'],
        'job_description': [
            'Join our amazing team! We offer excellent benefits, flexible work arrangements, and great growth opportunities.',
            'We are looking for a talented engineer to join our innovative team. Competitive salary and comprehensive benefits.',
            'Challenging role with tight deadlines and high pressure environment. Must work weekends and overtime.'
        ],
        'source': ['glassdoor', 'monster', 'naukri']
    })
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Generate sentiment report
    report = analyzer.generate_sentiment_report(sample_data)
    
    print("Sentiment Analysis Report:")
    print(f"Positive: {report['summary']['positive_percentage']:.1f}%")
    print(f"Negative: {report['summary']['negative_percentage']:.1f}%")
    print(f"Neutral: {report['summary']['neutral_percentage']:.1f}%")
