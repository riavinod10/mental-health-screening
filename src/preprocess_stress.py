"""
Preprocessing for Student Stress Factors Dataset
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def preprocess_stress_data():
    """
    Load and preprocess the student stress dataset
    """
    
    print("="*60)
    print("PREPROCESSING STRESS DATASET")
    print("="*60)
    
    # Load data
    df = pd.read_csv('data/raw/Student_Stress_Factors.csv')
    
    print(f"\n📊 Original dataset shape: {df.shape}")
    print(f"\n📋 Columns: {list(df.columns)}")
    
    # Rename columns for clarity
    column_mapping = {
        'Kindly Rate your Sleep Quality 😴': 'sleep_quality',
        'How many times a week do you suffer headaches 🤕?': 'headaches_weekly',
        'How would you rate you academic performance 👩‍🎓?': 'academic_performance',
        'how would you rate your study load?': 'study_load',
        'How many times a week you practice extracurricular activities 🎾?': 'extracurricular_weekly',
        'How would you rate your stress levels?': 'stress_level'
    }
    
    df = df.rename(columns=column_mapping)
    
    # Check for missing values
    print(f"\n🔍 Missing values:")
    print(df.isnull().sum())
    
    # Drop any rows with missing values (if any)
    initial_rows = len(df)
    df = df.dropna()
    print(f"\n🗑️ Dropped {initial_rows - len(df)} rows with missing values")
    
    # Check data types
    print(f"\n📊 Data types:")
    print(df.dtypes)
    
    # Convert all columns to numeric (just in case)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop any remaining NaN values after conversion
    df = df.dropna()
    
    # Define feature columns and target
    feature_cols = ['sleep_quality', 'headaches_weekly', 'academic_performance', 
                    'study_load', 'extracurricular_weekly']
    target_col = 'stress_level'
    
    # Create stress category (0=Low, 1=Moderate, 2=High)
    # Based on stress_level values (assuming 1-5 scale)
    def categorize_stress(level):
        if level <= 2:
            return 0  # Low
        elif level <= 3:
            return 1  # Moderate
        else:
            return 2  # High
    
    df['stress_category'] = df['stress_level'].apply(categorize_stress)
    
    # Feature statistics
    print(f"\n📈 Feature Statistics:")
    print(df[feature_cols].describe())
    
    print(f"\n📊 Stress Category Distribution:")
    print(df['stress_category'].value_counts())
    print(f"\nPercentage:")
    print(df['stress_category'].value_counts(normalize=True) * 100)
    
    # Save processed data
    df.to_csv('data/processed/stress_processed.csv', index=False)
    print(f"\n✅ Saved processed stress data to data/processed/stress_processed.csv")
    print(f"✅ Final dataset shape: {df.shape}")
    
    return df, feature_cols

if __name__ == "__main__":
    df, features = preprocess_stress_data()