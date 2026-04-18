"""
Mental Health Screening System - Main Entry Point
"""

import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the actual modules from your src folder
from src import utils
from src.preprocess_stress import preprocess_stress_data
from src.preprocess_depression import preprocess_depression_data
from src.train_models import train_stress_model, train_depression_model

def main():
    """Run the complete pipeline"""
    print("🚀 Starting Mental Health Screening Project\n")
    
    # Create folder structure if needed
    utils.create_folder_structure()
    
    # ========== STEP 1: PREPROCESSING ==========
    print("\n" + "="*70)
    print("STEP 1: PREPROCESSING")
    print("="*70)
    
    # Process stress dataset
    stress_df, stress_features = preprocess_stress_data()
    
    # Process depression dataset
    depression_df, depression_features = preprocess_depression_data()
    
    # ========== STEP 2: TRAINING ==========
    print("\n" + "="*70)
    print("STEP 2: MODEL TRAINING")
    print("="*70)
    
    # Train stress model
    stress_model, stress_scaler, stress_metrics = train_stress_model()
    
    # Train depression model
    depression_model, depression_scaler, depression_metrics = train_depression_model()
    
    # ========== SUMMARY ==========
    print("\n" + "="*70)
    print("✅ PROJECT COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nStress Model Accuracy:     {stress_metrics['accuracy']:.4f}")
    print(f"Depression Model Accuracy: {depression_metrics['accuracy']:.4f}")
    print("\n📁 Results saved in:")
    print("   - data/processed/    (cleaned datasets)")
    print("   - models/            (trained models & scalers)")
    print("   - results/figures/   (confusion matrices & feature importance)")
    print("   - results/           (metrics reports)")

if __name__ == "__main__":
    main()