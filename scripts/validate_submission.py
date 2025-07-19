"""
Submission Validation Script
Validates the generated submission file format and content
"""

import pandas as pd
import numpy as np

def validate_submission_file(filename: str = "best_submission.csv") -> bool:
    """
    Validate submission file format and content
    
    Args:
        filename: Path to submission file
        
    Returns:
        True if validation passes, False otherwise
    """
    print(f"Validating submission file: {filename}")
    
    try:
        # Load submission file
        submission = pd.read_csv(filename)
        
        # Check basic format
        print(f"✓ File loaded successfully")
        print(f"✓ Shape: {submission.shape}")
        
        # Check columns
        expected_columns = ['id', 'Personality']
        if list(submission.columns) != expected_columns:
            print(f"✗ Column mismatch. Expected: {expected_columns}, Got: {list(submission.columns)}")
            return False
        print(f"✓ Columns correct: {list(submission.columns)}")
        
        # Check data types
        if not pd.api.types.is_integer_dtype(submission['id']):
            print(f"✗ ID column should be integer, got: {submission['id'].dtype}")
            return False
        print(f"✓ ID column type correct: {submission['id'].dtype}")
        
        if submission['Personality'].dtype != 'object':
            print(f"✗ Personality column should be object, got: {submission['Personality'].dtype}")
            return False
        print(f"✓ Personality column type correct: {submission['Personality'].dtype}")
        
        # Check personality values
        valid_personalities = {'Introvert', 'Extrovert'}
        unique_personalities = set(submission['Personality'].unique())
        if unique_personalities != valid_personalities:
            print(f"✗ Invalid personality values. Expected: {valid_personalities}, Got: {unique_personalities}")
            return False
        print(f"✓ Personality values correct: {unique_personalities}")
        
        # Check for missing values
        if submission.isnull().any().any():
            print(f"✗ Missing values found:")
            print(submission.isnull().sum())
            return False
        print(f"✓ No missing values")
        
        # Check ID range and uniqueness
        if len(submission['id'].unique()) != len(submission):
            print(f"✗ Duplicate IDs found")
            return False
        print(f"✓ All IDs unique")
        
        # Expected test range: 18524 to 24698 (6175 samples)
        expected_min_id = 18524
        expected_max_id = 24698
        expected_count = 6175
        
        actual_min_id = submission['id'].min()
        actual_max_id = submission['id'].max()
        actual_count = len(submission)
        
        if actual_count != expected_count:
            print(f"✗ Wrong number of predictions. Expected: {expected_count}, Got: {actual_count}")
            return False
        print(f"✓ Correct number of predictions: {actual_count}")
        
        if actual_min_id != expected_min_id or actual_max_id != expected_max_id:
            print(f"✗ ID range mismatch. Expected: {expected_min_id}-{expected_max_id}, Got: {actual_min_id}-{actual_max_id}")
            return False
        print(f"✓ ID range correct: {actual_min_id}-{actual_max_id}")
        
        # Show prediction distribution
        personality_counts = submission['Personality'].value_counts()
        total = len(submission)
        print(f"\n📊 Prediction Distribution:")
        for personality, count in personality_counts.items():
            percentage = (count / total) * 100
            print(f"  {personality}: {count} ({percentage:.1f}%)")
        
        # Compare with expected CV training distribution (74.17% Extrovert, 25.83% Introvert)
        extrovert_pct = (personality_counts.get('Extrovert', 0) / total) * 100
        introvert_pct = (personality_counts.get('Introvert', 0) / total) * 100
        
        print(f"\n🎯 Expected from CV 0.9684 training:")
        print(f"  Extrovert: 74.17%")
        print(f"  Introvert: 25.83%")
        
        print(f"\n📈 Actual distribution:")
        print(f"  Extrovert: {extrovert_pct:.1f}%")
        print(f"  Introvert: {introvert_pct:.1f}%")
        
        # Check if distribution is reasonably close to CV training
        extrovert_diff = abs(extrovert_pct - 74.17)
        introvert_diff = abs(introvert_pct - 25.83)
        
        if extrovert_diff > 5 or introvert_diff > 5:
            print(f"⚠️  Distribution differs significantly from CV training (>5% difference)")
        else:
            print(f"✓ Distribution close to CV training (<5% difference)")
        
        print(f"\n✅ All validation checks passed!")
        return True
        
    except Exception as e:
        print(f"✗ Validation failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    validate_submission_file("best_submission.csv")