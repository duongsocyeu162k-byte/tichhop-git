#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Database Clear Script
====================

This script clears all data from PostgreSQL and MongoDB databases.
"""

import sys
import os
import yaml
import logging

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src")
sys.path.append(src_path)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """Load database configuration."""
    config_path = os.path.join(current_dir, "config", "config.yaml")
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config.get('database', {})
    except Exception as e:
        logger.error("Error loading config: {}".format(e))
        return {}

def clear_postgresql(config):
    """Clear PostgreSQL database."""
    logger.info("Clearing PostgreSQL database...")
    
    try:
        import psycopg2
        from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
        
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host=config['postgres']['host'],
            port=config['postgres']['port'],
            database=config['postgres']['database'],
            user=config['postgres']['username'],
            password=config['postgres']['password']
        )
        
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE'
        """)
        
        tables = cursor.fetchall()
        logger.info("Found {} tables in PostgreSQL".format(len(tables)))
        
        # Clear all tables
        for table in tables:
            table_name = table[0]
            try:
                cursor.execute("TRUNCATE TABLE {} CASCADE".format(table_name))
                logger.info("Cleared table: {}".format(table_name))
            except Exception as e:
                logger.warning("Error clearing table {}: {}".format(table_name, e))
        
        # Reset sequences
        cursor.execute("""
            SELECT sequence_name 
            FROM information_schema.sequences 
            WHERE sequence_schema = 'public'
        """)
        
        sequences = cursor.fetchall()
        for seq in sequences:
            seq_name = seq[0]
            try:
                cursor.execute("ALTER SEQUENCE {} RESTART WITH 1".format(seq_name))
                logger.info("Reset sequence: {}".format(seq_name))
            except Exception as e:
                logger.warning("Error resetting sequence {}: {}".format(seq_name, e))
        
        cursor.close()
        conn.close()
        logger.info("PostgreSQL database cleared successfully!")
        return True
        
    except ImportError:
        logger.warning("psycopg2 not installed. Skipping PostgreSQL.")
        return False
    except Exception as e:
        logger.error("Error clearing PostgreSQL: {}".format(e))
        return False

def clear_mongodb(config):
    """Clear MongoDB database."""
    logger.info("Clearing MongoDB database...")
    
    try:
        from pymongo import MongoClient
        
        # Connect to MongoDB
        client = MongoClient(
            host=config['mongodb']['host'],
            port=config['mongodb']['port'],
            username=config['mongodb']['username'],
            password=config['mongodb']['password']
        )
        
        db = client[config['mongodb']['database']]
        
        # Get all collections
        collections = db.list_collection_names()
        logger.info("Found {} collections in MongoDB".format(len(collections)))
        
        # Clear all collections
        for collection_name in collections:
            try:
                collection = db[collection_name]
                count = collection.count_documents({})
                collection.drop()
                logger.info("Cleared collection '{}' ({} documents)".format(collection_name, count))
            except Exception as e:
                logger.warning("Error clearing collection '{}': {}".format(collection_name, e))
        
        client.close()
        logger.info("MongoDB database cleared successfully!")
        return True
        
    except ImportError:
        logger.warning("pymongo not installed. Skipping MongoDB.")
        return False
    except Exception as e:
        logger.error("Error clearing MongoDB: {}".format(e))
        return False

def clear_analytics_metadata():
    """Clear analytics metadata files."""
    logger.info("Clearing analytics metadata...")
    
    metadata_files = [
        "output/processed_jobs_*.json",
        "logs/*.log",
        "*.log"
    ]
    
    import glob
    
    for pattern in metadata_files:
        files = glob.glob(os.path.join(current_dir, pattern))
        for file_path in files:
            try:
                os.remove(file_path)
                logger.info("Removed file: {}".format(file_path))
            except Exception as e:
                logger.warning("Error removing file {}: {}".format(file_path, e))

def main():
    """Main function."""
    logger.info("Starting database clear operation...")
    logger.info("=" * 50)
    
    # Load configuration
    config = load_config()
    if not config:
        logger.error("Failed to load configuration. Exiting.")
        return
    
    # Clear PostgreSQL
    postgres_success = clear_postgresql(config)
    
    # Clear MongoDB
    mongodb_success = clear_mongodb(config)
    
    # Clear analytics metadata
    clear_analytics_metadata()
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("DATABASE CLEAR SUMMARY")
    logger.info("=" * 50)
    
    if postgres_success:
        logger.info("PostgreSQL: CLEARED ✅")
    else:
        logger.info("PostgreSQL: SKIPPED ⚠️")
    
    if mongodb_success:
        logger.info("MongoDB: CLEARED ✅")
    else:
        logger.info("MongoDB: SKIPPED ⚠️")
    
    logger.info("Analytics metadata: CLEARED ✅")
    
    if postgres_success or mongodb_success:
        logger.info("\nDatabase clear operation completed successfully! 🗑️")
    else:
        logger.info("\nNo databases were cleared. Check your configuration and dependencies.")

if __name__ == "__main__":
    main()
