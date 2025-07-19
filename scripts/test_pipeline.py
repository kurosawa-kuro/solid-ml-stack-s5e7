#!/usr/bin/env python3
"""
Test script for bronze → silver → gold data pipeline
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.bronze import load_data
from src.data.silver import DataPipeline
from src.data.gold import DataManager

def main():
    print("🔄 Testing Bronze → Silver → Gold Pipeline")
    print("=" * 50)
    
    # Bronze: Load raw data
    print("\n📥 BRONZE: Loading raw data...")
    train_raw, test_raw = load_data()
    print(f"✅ Train shape: {train_raw.shape}")
    print(f"✅ Test shape: {test_raw.shape}")
    print(f"✅ Columns: {list(train_raw.columns)}")
    
    # Silver: Basic preprocessing
    print("\n⚙️  SILVER: Basic preprocessing...")
    db_path = os.getenv('DB_PATH', '/home/wsl/dev/my-study/ml/solid-ml-stack-s5e7/data/kaggle_datasets.duckdb')
    pipeline = DataPipeline(db_path)
    train_clean = pipeline.preprocess(train_raw)
    test_clean = pipeline.preprocess(test_raw)
    print(f"✅ Train clean shape: {train_clean.shape}")
    print(f"✅ Test clean shape: {test_clean.shape}")
    print(f"✅ Dtypes: {dict(train_clean.dtypes)}")
    
    # Gold: Feature engineering  
    print("\n✨ GOLD: Feature engineering...")
    dm = DataManager()
    X_train, X_test = dm.get_data()
    y_train = train_clean['Personality']
    print(f"✅ X_train shape: {X_train.shape}")
    print(f"✅ y_train shape: {y_train.shape}")
    print(f"✅ X_test shape: {X_test.shape}")
    print(f"✅ Feature columns: {list(X_train.columns)}")
    
    # Summary
    print("\n📊 PIPELINE SUMMARY")
    print("=" * 50)
    print(f"Raw → Clean: {train_raw.shape} → {train_clean.shape}")
    print(f"Clean → Features: {train_clean.shape} → {X_train.shape}")
    print(f"Target distribution: {y_train.value_counts().to_dict()}")
    print("\n✅ Pipeline test completed successfully!")

if __name__ == "__main__":
    main()