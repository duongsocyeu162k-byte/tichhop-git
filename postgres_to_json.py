#!/usr/bin/env python3
"""
File duy nhất để đọc PostgreSQL và ghi sang JSON
Sử dụng: python postgres_to_json.py
"""

import json
import psycopg2
from psycopg2.extras import RealDictCursor
import yaml
from datetime import datetime
import os


def load_config():
    """Load cấu hình database từ config.yaml"""
    with open('config/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config['database']['postgres']


def export_table(table_name, batch_size=1000, limit=None):
    """
    Export bảng PostgreSQL sang JSON theo batch
    
    Args:
        table_name: Tên bảng cần export
        batch_size: Kích thước mỗi batch (mặc định 1000)
        limit: Giới hạn số records (None = tất cả)
    """
    print(f"🔄 Exporting table '{table_name}'...")
    
    # Load config và kết nối database
    config = load_config()
    conn = psycopg2.connect(
        host=config['host'],
        port=config['port'],
        database=config['database'],
        user=config['username'],
        password=config['password']
    )
    
    # Tạo thư mục output
    os.makedirs('output', exist_ok=True)
    
    try:
        # Lấy tổng số records
        with conn.cursor() as cursor:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            total_records = cursor.fetchone()[0]
            
        if limit:
            total_records = min(total_records, limit)
            
        print(f"📊 Total records: {total_records:,}")
        print(f"📦 Batch size: {batch_size:,}")
        
        # Tạo tên file với timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"output/{table_name}_{timestamp}.json"
        
        all_data = []
        offset = 0
        batch_count = 0
        
        # Export theo batch
        while offset < total_records:
            batch_count += 1
            current_batch_size = min(batch_size, total_records - offset)
            
            print(f"📦 Batch {batch_count}: {offset+1:,} - {offset+current_batch_size:,}")
            
            # Query batch data
            query = f"""
                SELECT * FROM {table_name} 
                ORDER BY id 
                LIMIT {current_batch_size} OFFSET {offset}
            """
            
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
                batch_data = [dict(row) for row in rows]
                all_data.extend(batch_data)
                
                print(f"   ✅ Loaded {len(batch_data):,} records")
            
            offset += current_batch_size
            
            # Progress
            progress = (offset / total_records) * 100
            print(f"   📈 Progress: {progress:.1f}%")
        
        # Ghi file JSON
        print(f"💾 Writing {len(all_data):,} records to JSON...")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✅ Success! File: {filename}")
        print(f"📊 Summary: {len(all_data):,} records in {batch_count} batches")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        conn.close()


def list_tables():
    """Liệt kê các bảng trong database"""
    config = load_config()
    conn = psycopg2.connect(
        host=config['host'],
        port=config['port'],
        database=config['database'],
        user=config['username'],
        password=config['password']
    )
    
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_type = 'BASE TABLE'
                ORDER BY table_name;
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            print(f"📋 Found {len(tables)} tables:")
            for i, table in enumerate(tables, 1):
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"  {i:2d}. {table:<25} ({count:,} records)")
            
            return tables
    finally:
        conn.close()


if __name__ == "__main__":
    import sys
    
    print("🚀 PostgreSQL to JSON Exporter")
    print("=" * 40)
    
    # Kiểm tra kết nối
    try:
        config = load_config()
        conn = psycopg2.connect(
            host=config['host'],
            port=config['port'],
            database=config['database'],
            user=config['username'],
            password=config['password']
        )
        conn.close()
        print("✅ Database connection successful!")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        exit(1)
    
    # Liệt kê bảng
    tables = list_tables()
    
    if not tables:
        print("❌ No tables found!")
        exit(1)
    
    # Nếu có tham số dòng lệnh
    if len(sys.argv) > 1:
        try:
            table_idx = int(sys.argv[1]) - 1
            if 0 <= table_idx < len(tables):
                table_name = tables[table_idx]
                batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
                limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
                
                print(f"🔄 Auto-exporting table: {table_name}")
                export_table(table_name, batch_size, limit)
            else:
                print(f"❌ Invalid table index! Use 1-{len(tables)}")
        except ValueError:
            print("❌ Invalid arguments! Usage: python postgres_to_json.py [table_index] [batch_size] [limit]")
    else:
        # Menu tương tác
        print(f"\n📋 Choose table to export:")
        for i, table in enumerate(tables, 1):
            print(f"  {i}. {table}")
        
        try:
            choice = int(input(f"\nEnter choice (1-{len(tables)}): ")) - 1
            if 0 <= choice < len(tables):
                table_name = tables[choice]
                
                # Hỏi batch size và limit
                batch_size = input("Batch size (default 1000): ").strip()
                batch_size = int(batch_size) if batch_size.isdigit() else 1000
                
                limit = input("Limit records (Enter for all): ").strip()
                limit = int(limit) if limit.isdigit() else None
                
                # Export
                export_table(table_name, batch_size, limit)
            else:
                print("❌ Invalid choice!")
        except ValueError:
            print("❌ Please enter a number!")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
        except EOFError:
            print("\n💡 Tip: Use command line arguments for non-interactive mode:")
            print("   python postgres_to_json.py 2 1000 5000")
            print("   (table_index=2, batch_size=1000, limit=5000)")
