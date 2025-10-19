#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETL Output Format Validation Script
==================================

This script validates the output format of the ETL pipeline to ensure
data is properly standardized and unified.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from etl.data_loader import DataLoader
from etl.data_cleaner import DataCleaner

def validate_output_format():
    """Validate the output format of ETL pipeline."""
    print("ETL Output Format Validation")
    print("=" * 50)
    
    try:
        # Load data
        loader = DataLoader()
        raw_data = loader.load_all_sources()
        
        if not any(not df.empty for df in raw_data.values()):
            print("FAIL: No data loaded. Please check data files.")
            return
        
        # Clean data
        cleaner = DataCleaner()
        cleaned_data = cleaner.clean_all_data(raw_data)
        standardized_data = cleaner.standardize_columns(cleaned_data)
        
        # Define expected standard columns
        expected_columns = [
            'source', 'job_title_clean', 'company_name', 'location_clean',
            'city', 'state', 'country', 'salary_min', 'salary_max',
            'industry', 'job_description', 'rating', 'company_size',
            'skills', 'experience'
        ]
        
        print("Standard Column Validation:")
        print("Expected columns: {}".format(expected_columns))
        print()
        
        # Validate each data source
        for source, df in standardized_data.items():
            if df.empty:
                print("{}: No data".format(source.upper()))
                continue
                
            print("{} Data Validation:".format(source.upper()))
            print("  Rows: {:,}".format(len(df)))
            print("  Columns: {}".format(len(df.columns)))
            
            # Check column presence
            missing_columns = [col for col in expected_columns if col not in df.columns]
            extra_columns = [col for col in df.columns if col not in expected_columns]
            
            if missing_columns:
                print("  ❌ Missing columns: {}".format(missing_columns))
            else:
                print("  ✅ All expected columns present")
            
            if extra_columns:
                print("  ⚠️  Extra columns: {}".format(extra_columns))
            
            # Check data types
            print("  Data Types:")
            for col in expected_columns:
                if col in df.columns:
                    dtype = str(df[col].dtype)
                    non_null_count = df[col].notna().sum()
                    print("    {}: {} ({} non-null)".format(col, dtype, non_null_count))
            
            # Check data quality
            quality_score = cleaner.get_data_quality_score(df)
            print("  Data Quality Score: {}/100".format(quality_score))
            
            # Sample data
            print("  Sample Data (first 2 rows):")
            sample_cols = ['job_title_clean', 'company_name', 'location_clean', 'salary_min', 'salary_max', 'skills', 'experience']
            available_cols = [col for col in sample_cols if col in df.columns]
            if available_cols:
                sample_df = df[available_cols].head(2)
                print(sample_df.to_string(index=False))
            
            print()
        
        # Check data unification
        print("Data Unification Check:")
        all_sources_have_data = all(not df.empty for df in standardized_data.values())
        if all_sources_have_data:
            print("✅ All data sources have standardized data")
            
            # Check if we can combine data
            non_empty_dfs = [df for df in standardized_data.values() if not df.empty]
            if non_empty_dfs:
                # Ensure all DataFrames have the same columns
                all_columns = set()
                for df in non_empty_dfs:
                    all_columns.update(df.columns)
                
                # Add missing columns to each DataFrame
                for df in non_empty_dfs:
                    for col in all_columns:
                        if col not in df.columns:
                            df[col] = None
                
                combined_df = pd.concat(non_empty_dfs, ignore_index=True)
                print("✅ Data can be combined into unified format")
                print("  Combined rows: {:,}".format(len(combined_df)))
                print("  Combined columns: {}".format(len(combined_df.columns)))
                
                # Show sample of combined data
                print("  Sample Combined Data:")
                sample_cols = ['source', 'job_title_clean', 'company_name', 'location_clean', 'salary_min', 'salary_max']
                available_cols = [col for col in sample_cols if col in combined_df.columns]
                if available_cols:
                    sample_combined = combined_df[available_cols].head(3)
                    print(sample_combined.to_string(index=False))
        else:
            print("❌ Some data sources are empty")
        
        print()
        
        # Test comprehensive analysis
        print("Testing Comprehensive Analysis:")
        try:
            comprehensive_analysis = cleaner.get_comprehensive_analysis(raw_data, standardized_data)
            print("✅ Comprehensive analysis completed")
            print("  Overall Quality Score: {}/100".format(comprehensive_analysis['overall_quality_score']))
            
            # Schema analysis
            schema_analysis = comprehensive_analysis['schema_analysis']
            print("  Schema Compatibility: {:.2%}".format(
                schema_analysis['compatibility']['overall_compatibility']
            ))
            
            # Matching analysis
            matching_analysis = comprehensive_analysis['matching_analysis']
            if 'matching_report' in matching_analysis:
                matching_report = matching_analysis['matching_report']
                print("  Duplicate Rate: {:.2%}".format(
                    matching_report['duplicate_analysis']['duplicate_rate']
                ))
                print("  Similarity Rate: {:.2%}".format(
                    matching_report['similarity_analysis']['similarity_rate']
                ))
        except Exception as e:
            print("❌ Comprehensive analysis failed: {}".format(e))
        
        print()
        print("✅ ETL Output Format Validation Completed!")
        
    except Exception as e:
        print("❌ Validation failed: {}".format(e))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    validate_output_format()
