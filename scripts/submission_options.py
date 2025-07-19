#!/usr/bin/env python3
"""
Show all available submission generation options
"""

import os
from pathlib import Path

def show_submission_options():
    """Display all available ways to generate submissions"""
    
    print("=" * 60)
    print("SUBMISSION GENERATION OPTIONS")
    print("=" * 60)
    
    print("\n🚀 QUICK SOLUTIONS (for immediate submission):")
    print("-" * 50)
    
    print("\n1. Simple Prediction (Handles nan/inf issues)")
    print("   Command: make simple-predict")
    print("   Output:  submissions/simple_submission.csv")
    print("   CV Score: ~0.9689")
    print("   Features: 15 basic features with data cleaning")
    print("   Time: ~30 seconds")
    
    print("\n2. Enhanced Prediction (Gold layer features)")
    print("   Command: make enhanced-predict")
    print("   Output:  submissions/enhanced_gold_submission.csv")
    print("   CV Score: ~0.9676")
    print("   Features: 50 engineered features from medallion architecture")
    print("   Time: ~60 seconds")
    
    print("\n📊 ADVANCED SOLUTIONS (for optimization):")
    print("-" * 50)
    
    print("\n3. Silver Enhanced Training")
    print("   Command: make train-silver-enhanced")
    print("   Output:  submissions/enhanced_silver_submission.csv (if CV > 0.975)")
    print("   Target:  Bronze medal (0.976518)")
    print("   Features: 100+ features with advanced preprocessing")
    print("   Time: ~5 minutes")
    
    print("\n🛠️ DIRECT SCRIPTS:")
    print("-" * 50)
    
    print("\n• python3 scripts/simple_predict.py")
    print("• PYTHONPATH=. python3 scripts/predict_with_gold.py")
    print("• python scripts/train_silver_enhanced.py")
    
    # Check existing submissions
    submissions_dir = Path("submissions")
    if submissions_dir.exists():
        submissions = list(submissions_dir.glob("*.csv"))
        if submissions:
            print(f"\n📁 EXISTING SUBMISSIONS ({len(submissions)} files):")
            print("-" * 50)
            for sub_file in sorted(submissions):
                size = sub_file.stat().st_size
                lines = sum(1 for _ in open(sub_file)) if sub_file.exists() else 0
                print(f"   {sub_file.name}: {size:,} bytes, {lines:,} lines")
    
    print("\n🎯 RECOMMENDATION:")
    print("-" * 50)
    print("For immediate submission: make simple-predict")
    print("For best performance:     make enhanced-predict")
    print("For bronze medal attempt: make train-silver-enhanced")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    show_submission_options()