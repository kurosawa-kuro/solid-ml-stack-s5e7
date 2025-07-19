#!/usr/bin/env python3
"""
Test script for bronze → silver → gold data pipeline
"""
import os
import sys

from src.data.bronze import load_data
from src.data.silver import create_silver_tables, load_silver_data

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def main():
    print("🔄 Testing Bronze → Silver → Gold Pipeline")
    print("=" * 50)

    # Bronze: Load raw data
    print("\n📥 BRONZE: Loading raw data...")
    train_raw, test_raw = load_data()
    print(f"✅ Train shape: {train_raw.shape}")
    print(f"✅ Test shape: {test_raw.shape}")
    print(f"✅ Columns: {list(train_raw.columns)}")

    # Silver: Create silver tables
    print("\n⚙️  SILVER: Creating silver tables...")
    create_silver_tables()
    train_silver, test_silver = load_silver_data()
    print(f"✅ Train silver shape: {train_silver.shape}")
    print(f"✅ Test silver shape: {test_silver.shape}")
    print(f"✅ Dtypes: {dict(train_silver.dtypes)}")

    # Gold: Feature engineering
    print("\n✨ GOLD: Feature engineering...")
    # TODO: Implement gold layer
    X_train = train_silver.drop("Personality", axis=1, errors="ignore")
    X_test = test_silver
    y_train = train_silver["Personality"] if "Personality" in train_silver.columns else None
    print(f"✅ X_train shape: {X_train.shape}")
    print(f"✅ y_train shape: {y_train.shape if y_train is not None else 'None'}")
    print(f"✅ X_test shape: {X_test.shape}")
    print(f"✅ Feature columns: {list(X_train.columns)}")

    # Summary
    print("\n📊 PIPELINE SUMMARY")
    print("=" * 50)
    print(f"Raw → Silver: {train_raw.shape} → {train_silver.shape}")
    print(f"Silver → Features: {train_silver.shape} → {X_train.shape}")
    if y_train is not None:
        print(f"Target distribution: {y_train.value_counts().to_dict()}")
    print("\n✅ Pipeline test completed successfully!")


if __name__ == "__main__":
    main()
