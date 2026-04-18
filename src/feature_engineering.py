"""
Feature Engineering for both datasets
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def engineer_stress_features(df):
    """
    Create additional features for stress dataset
    """
    print("\n" + "="*60)
    print("FEATURE ENGINEERING - STRESS DATASET")
    print("="*60)
    
    # Create interaction features
    # Sleep quality and headaches interaction
    df['sleep_headache_interaction'] = df['sleep_quality'] * df['headaches_weekly']
    
    # Academic load ratio (study load vs academic performance)
    df['academic_load_ratio'] = df['study_load'] / (df['academic_performance'] + 0.01)
    
    # Stress risk score (weighted combination)
    df['stress_risk_score'] = (
        df['sleep_quality'] * 0.3 +
        df['headaches_weekly'] * 0.25 +
        df['study_load'] * 0.25 +
        (5 - df['academic_performance']) * 0.1 +
        (5 - df['extracurricular_weekly']) * 0.1
    )
    
    # Binned features
    df['sleep_category'] = pd.cut(df['sleep_quality'], bins=[0, 2, 4, 5], labels=[0, 1, 2]).astype(int)
    df['headache_category'] = pd.cut(df['headaches_weekly'], bins=[-1, 1, 3, 5], labels=[0, 1, 2]).astype(int)
    
    # Get final feature list
    original_features = ['sleep_quality', 'headaches_weekly', 'academic_performance', 
                         'study_load', 'extracurricular_weekly']
    new_features = ['sleep_headache_interaction', 'academic_load_ratio', 
                    'stress_risk_score', 'sleep_category', 'headache_category']
    
    all_features = original_features + new_features
    
    print(f"✅ Created {len(new_features)} new features")
    print(f"✅ Total features: {len(all_features)}")
    print(f"\nNew features created:")
    for f in new_features:
        print(f"  - {f}")
    
    return df, all_features

def engineer_depression_features(df, feature_cols):
    """
    Create additional features for depression dataset
    """
    print("\n" + "="*60)
    print("FEATURE ENGINEERING - DEPRESSION DATASET")
    print("="*60)
    
    # Create composite scores
    # Mood cluster (interest, feeling down, feeling bad)
    mood_features = ['little_interest_score', 'feeling_down_score', 'feeling_bad_score']
    if all(f in df.columns for f in mood_features):
        df['mood_cluster_score'] = df[mood_features].sum(axis=1)
    
    # Energy cluster (sleep, energy, concentration)
    energy_features = ['sleep_issues_score', 'low_energy_score', 'concentration_issues_score']
    if all(f in df.columns for f in energy_features):
        df['energy_cluster_score'] = df[energy_features].sum(axis=1)
    
    # Physical cluster (appetite, movement)
    physical_features = ['appetite_changes_score', 'moving_slowly_score']
    if all(f in df.columns for f in physical_features):
        df['physical_cluster_score'] = df[physical_features].sum(axis=1)
    
    # Risk indicator (self-harm presence)
    if 'self_harm_score' in df.columns:
        df['high_risk_flag'] = (df['self_harm_score'] >= 2).astype(int)
    
    # Get final feature list
    original_features = feature_cols
    new_features = [f for f in ['mood_cluster_score', 'energy_cluster_score', 
                                 'physical_cluster_score', 'high_risk_flag'] 
                    if f in df.columns]
    
    all_features = original_features + new_features
    
    print(f"✅ Created {len(new_features)} new features")
    print(f"✅ Total features: {len(all_features)}")
    print(f"\nNew features created:")
    for f in new_features:
        if f in df.columns:
            print(f"  - {f}")
    
    return df, all_features

def scale_features(X_train, X_test, feature_names):
    """
    Standardize features
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for easier interpretation
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)
    
    print(f"\n✅ Features scaled using StandardScaler")
    print(f"   Training set shape: {X_train_scaled.shape}")
    print(f"   Test set shape: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, scaler

if __name__ == "__main__":
    # Test the functions
    stress_df = pd.read_csv('data/processed/stress_processed.csv')
    depression_df = pd.read_csv('data/processed/depression_processed.csv')
    
    stress_df, stress_features = engineer_stress_features(stress_df)
    depression_df, depression_features = engineer_depression_features(
        depression_df, 
        [col for col in depression_df.columns if col.endswith('_score')]
    )
    
    print("\n✅ Feature engineering completed!")