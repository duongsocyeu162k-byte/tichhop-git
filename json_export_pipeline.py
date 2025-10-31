#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON Export Pipeline
====================

Pipeline đơn giản để trích xuất, làm sạch và xuất dữ liệu ra file JSON.
"""

import sys
import os
import pandas as pd
import json
import logging
from datetime import datetime
from typing import Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from etl.data_loader import DataLoader
from etl.data_cleaner import DataCleaner
from analytics.trend_analyzer import TrendAnalyzer
from analytics.comprehensive_analyzer import ComprehensiveAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class JSONExporter:
    """Xuất dữ liệu đã xử lý ra file JSON."""
    
    def __init__(self, output_dir: str = "output"):
        """
        Khởi tạo JSONExporter.
        
        Args:
            output_dir: Thư mục để lưu file JSON
        """
        self.output_dir = output_dir
        
        # Tạo thư mục output nếu chưa tồn tại
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")
    
    def export_processed_data(self, data: pd.DataFrame, filename: str = None) -> str:
        """
        Xuất dữ liệu đã xử lý ra file JSON.
        
        Args:
            data: DataFrame chứa dữ liệu đã xử lý
            filename: Tên file (optional)
            
        Returns:
            str: Đường dẫn đến file đã lưu
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"processed_jobs_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            # Convert DataFrame to dict
            data_dict = data.to_dict('records')
            
            # Write to JSON file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data_dict, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"Exported {len(data_dict)} records to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to export data: {e}")
            raise
    
    def export_analytics_report(self, report: Dict[str, Any], filename: str = None) -> str:
        """
        Xuất báo cáo phân tích ra file JSON.
        
        Args:
            report: Dictionary chứa kết quả phân tích
            filename: Tên file (optional)
            
        Returns:
            str: Đường dẫn đến file đã lưu
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analytics_report_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"Exported analytics report to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to export analytics report: {e}")
            raise
    
    def export_summary(self, summary: Dict[str, Any], filename: str = "pipeline_summary.json") -> str:
        """
        Xuất tóm tắt pipeline ra file JSON.
        
        Args:
            summary: Dictionary chứa tóm tắt
            filename: Tên file
            
        Returns:
            str: Đường dẫn đến file đã lưu
        """
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"Exported pipeline summary to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to export summary: {e}")
            raise


def main():
    """Hàm main để chạy pipeline."""
    print("=" * 60)
    print("JSON Export Pipeline - Job Market Analytics")
    print("=" * 60)
    print()
    
    try:
        # 1. Load dữ liệu
        print("1. Đang tải dữ liệu từ các nguồn...")
        loader = DataLoader()
        raw_data = loader.load_all_sources()
        
        total_raw_records = sum(len(df) for df in raw_data.values())
        print(f"   ✓ Đã tải {total_raw_records:,} records từ {len(raw_data)} nguồn")
        for source, df in raw_data.items():
            print(f"     - {source}: {len(df):,} records")
        print()
        
        # 2. Làm sạch và chuẩn hóa dữ liệu
        print("2. Đang làm sạch và chuẩn hóa dữ liệu...")
        cleaner = DataCleaner()  # Không cần MongoDB nữa
        cleaned_data = cleaner.clean_all_data(raw_data)
        standardized_data = cleaner.standardize_columns(cleaned_data)
        
        total_cleaned_records = sum(len(df) for df in standardized_data.values())
        print(f"   ✓ Đã xử lý {total_cleaned_records:,} records")
        print()
        
        # 3. Kết hợp dữ liệu từ tất cả nguồn
        print("3. Đang kết hợp dữ liệu...")
        all_data = []
        for source, df in standardized_data.items():
            if not df.empty:
                df_copy = df.copy()
                df_copy['source'] = source
                all_data.append(df_copy)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"   ✓ Đã kết hợp thành {len(combined_df):,} records")
            
            # Xóa các cột không cần thiết
            columns_to_remove = ['original_id', 'sector', 'sentiment_score']
            existing_columns_to_remove = [col for col in columns_to_remove if col in combined_df.columns]
            if existing_columns_to_remove:
                combined_df = combined_df.drop(columns=existing_columns_to_remove)
                print(f"   ✓ Đã xóa các cột: {', '.join(existing_columns_to_remove)}")
        else:
            print("   ✗ Không có dữ liệu để xử lý")
            return
        print()
        
        # 4. Tạo báo cáo phân tích
        print("4. Đang tạo báo cáo phân tích...")
        comprehensive_analyzer = ComprehensiveAnalyzer()
        analytics_report = comprehensive_analyzer.generate_comprehensive_report(combined_df)
        
        print(f"   ✓ Đã tạo báo cáo phân tích toàn diện")
        print()
        
        # 5. Xuất dữ liệu ra file JSON
        print("5. Đang xuất dữ liệu ra file JSON...")
        exporter = JSONExporter(output_dir="output")
        
        # Xuất dữ liệu đã xử lý
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        processed_file = exporter.export_processed_data(
            combined_df,
            filename=f"processed_jobs_{timestamp}.json"
        )
        print(f"   ✓ Dữ liệu đã xử lý: {processed_file}")
        
        # Xuất báo cáo phân tích
        analytics_file = exporter.export_analytics_report(
            analytics_report,
            filename=f"analytics_report_{timestamp}.json"
        )
        print(f"   ✓ Báo cáo phân tích: {analytics_file}")
        
        # Tạo và xuất tóm tắt pipeline
        pipeline_summary = {
            'pipeline_info': {
                'run_date': datetime.now().isoformat(),
                'pipeline_version': '2.0 (JSON Export)',
                'output_directory': 'output/'
            },
            'data_sources': {
                source: {
                    'raw_records': len(raw_data[source]),
                    'processed_records': len(standardized_data[source])
                }
                for source in raw_data.keys()
            },
            'statistics': {
                'total_raw_records': total_raw_records,
                'total_processed_records': len(combined_df),
                'unique_companies': combined_df['company_name'].nunique() if 'company_name' in combined_df.columns else 0,
                'unique_locations': combined_df['city'].nunique() if 'city' in combined_df.columns else 0,
            },
            'files_generated': {
                'processed_data': processed_file,
                'analytics_report': analytics_file
            }
        }
        
        summary_file = exporter.export_summary(pipeline_summary)
        print(f"   ✓ Tóm tắt pipeline: {summary_file}")
        print()
        
        # 6. Hiển thị thống kê
        print("=" * 60)
        print("Tóm tắt Pipeline")
        print("=" * 60)
        print(f"Tổng số records gốc:      {total_raw_records:,}")
        print(f"Tổng số records đã xử lý:  {len(combined_df):,}")
        print(f"Số lượng công ty:          {pipeline_summary['statistics']['unique_companies']:,}")
        print(f"Số lượng địa điểm:         {pipeline_summary['statistics']['unique_locations']:,}")
        print()
        
        # Hiển thị key metrics từ analytics
        if 'analytics_summary' in analytics_report:
            summary = analytics_report['analytics_summary']
            print("Kết quả phân tích:")
            print("-" * 40)
            
            # Modules status
            if 'modules_status' in summary:
                print("Trạng thái các module:")
                for module, status in summary['modules_status'].items():
                    status_icon = "✓" if status == 'success' else "✗"
                    print(f"  {status_icon} {module.replace('_', ' ').title()}")
                print()
            
            # Key metrics
            if 'key_metrics' in summary:
                metrics = summary['key_metrics']
                print("Các chỉ số chính:")
                if 'anomaly_rate' in metrics:
                    print(f"  • Tỷ lệ bất thường:       {metrics['anomaly_rate']:.1f}%")
                if 'fraud_rate' in metrics:
                    print(f"  • Tỷ lệ gian lận:         {metrics['fraud_rate']:.1f}%")
                if 'positive_sentiment' in metrics:
                    print(f"  • Cảm xúc tích cực:       {metrics['positive_sentiment']:.1f}%")
                if 'salary_model_performance' in metrics:
                    print(f"  • Hiệu suất mô hình lương: {metrics['salary_model_performance']:.3f}")
                print()
        
        print("=" * 60)
        print("✓ Pipeline hoàn tất thành công!")
        print(f"✓ Các file đã được lưu trong thư mục: {exporter.output_dir}/")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Pipeline thất bại: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

