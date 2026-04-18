"""
Preprocessing for PHQ-9 Depression Dataset
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def preprocess_depression_data():
    """
    Load and preprocess the PHQ-9 depression dataset
    """
    
    print("="*60)
    print("PREPROCESSING DEPRESSION DATASET (PHQ-9)")
    print("="*60)
    
    # Load Excel file
    df = pd.read_excel('data/raw/PHQ9_Student_Depression_Dataset_Updated.xlsx')
    
    print(f"\n📊 Original dataset shape: {df.shape}")
    
    # Rename columns for clarity
    column_mapping = {
        'Do you have little interest or pleasure in doing things?': 'little_interest',
        'Do you feel down, depressed, or hopeless?': 'feeling_down',
        'Do you have trouble falling or staying asleep, or do you sleep too much?': 'sleep_issues',
        'Do you feel tired or have little energy?': 'low_energy',
        'Do you have poor appetite or tend to overeat?': 'appetite_changes',
        'Do you feel bad about yourself or that you are a failure or have let yourself or your family down?': 'feeling_bad',
        'Do you have trouble concentrating on things, such as reading, work, or watching television?': 'concentration_issues',
        'Have you been moving or speaking so slowly that other people have noticed, or the opposite—being fidgety or restless?': 'moving_slowly',
        'Have you had thoughts of self-harm or felt that you would be better off dead?': 'self_harm'
    }
    
    # Rename only the columns that exist
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})
    
    print(f"\n📋 Columns: {list(df.columns)}")
    
    # Encode text responses to numeric scores
    # Mapping for severity levels
    severity_mapping = {
        'Minimal': 0,
        'Mild': 1,
        'Moderate': 2,
        'Moderately Severe': 3,
        'Severe': 4
    }
    
    # Map severity levels
    if 'Severity Level' in df.columns:
        df['severity_encoded'] = df['Severity Level'].map(severity_mapping)
    
    # We need to convert the text responses to numeric scores (0-3)
    # Let's create a scoring function based on the PHQ-9 standard
    # Each response corresponds to a score: 0, 1, 2, or 3
    
    # PHQ-9 scoring patterns
    def get_phq9_score(response_text):
        response_text = str(response_text).lower()
        
        # Score 0 - Minimal/No symptoms
        if any(word in response_text for word in ['no thoughts', 'no noticeable', 'no major changes', 
                                                   'normal', 'fine', 'good', 'easily', 'confident',
                                                   'can focus', 'sleeping well', 'eating habits normal']):
            return 0
        
        # Score 1 - Mild/Several days
        elif any(word in response_text for word in ['slight trouble', 'occasionally', 'a little more tired',
                                                      'sometimes feel down', 'fidget a little', 'distracted easily',
                                                      'appetite fluctuates', 'not every night']):
            return 1
        
        # Score 2 - Moderate/More than half the days
        elif any(word in response_text for word in ['more than half', 'struggle to concentrate', 
                                                      'feeling down most days', 'changed significantly',
                                                      'some thoughts', 'wouldn’t act', 'disappearing']):
            return 2
        
        # Score 3 - Severe/Nearly every day
        elif any(word in response_text for word in ['constantly feel', 'always exhausted', 'every day',
                                                      'barely sleep', 'nightmares', 'no motivation',
                                                      'can’t concentrate', 'worthless', 'end everything',
                                                      'self-harm', 'completely lost']):
            return 3
        
        else:
            return 0  # Default to minimal
    
    # Apply scoring to all PHQ-9 question columns
    phq9_columns = ['little_interest', 'feeling_down', 'sleep_issues', 'low_energy',
                    'appetite_changes', 'feeling_bad', 'concentration_issues', 
                    'moving_slowly', 'self_harm']
    
    for col in phq9_columns:
        if col in df.columns:
            df[f'{col}_score'] = df[col].apply(get_phq9_score)
            print(f"✅ Encoded {col}")
    
    # Calculate total PHQ-9 score from encoded scores
    score_columns = [f'{col}_score' for col in phq9_columns if f'{col}_score' in df.columns]
    if score_columns:
        df['phq9_calculated'] = df[score_columns].sum(axis=1)
    
    # Use provided PHQ-9 Score if available, otherwise use calculated
    if 'PHQ-9 Score' in df.columns:
        df['phq9_final'] = df['PHQ-9 Score']
    else:
        df['phq9_final'] = df['phq9_calculated']
    
    # Create severity categories from scores
    def score_to_severity(score):
        if score <= 4:
            return 0  # Minimal
        elif score <= 9:
            return 1  # Mild
        elif score <= 14:
            return 2  # Moderate
        elif score <= 19:
            return 3  # Moderately Severe
        else:
            return 4  # Severe
    
    df['severity_from_score'] = df['phq9_final'].apply(score_to_severity)
    
    # Use encoded severity if available, otherwise use score-based
    if 'severity_encoded' in df.columns:
        df['target'] = df['severity_encoded']
    else:
        df['target'] = df['severity_from_score']
    
    # Feature columns for ML
    feature_cols = [f'{col}_score' for col in phq9_columns if f'{col}_score' in df.columns]
    
    print(f"\n📊 PHQ-9 Score Statistics:")
    print(f"Min: {df['phq9_final'].min()}")
    print(f"Max: {df['phq9_final'].max()}")
    print(f"Mean: {df['phq9_final'].mean():.2f}")
    
    print(f"\n📊 Depression Severity Distribution:")
    severity_names = {0: 'Minimal', 1: 'Mild', 2: 'Moderate', 3: 'Moderately Severe', 4: 'Severe'}
    severity_counts = df['target'].value_counts().sort_index()
    for severity, count in severity_counts.items():
        print(f"  {severity_names.get(severity, severity)}: {count} ({count/len(df)*100:.1f}%)")
    
    # Save processed data
    df.to_csv('data/processed/depression_processed.csv', index=False)
    print(f"\n✅ Saved processed depression data to data/processed/depression_processed.csv")
    print(f"✅ Final dataset shape: {df.shape}")
    
    return df, feature_cols

if __name__ == "__main__":
    df, features = preprocess_depression_data()