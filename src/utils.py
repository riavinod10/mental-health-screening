"""
Utility functions for mental health screening project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import os

def create_folder_structure():
    """Create all necessary folders"""
    folders = ['data/raw', 'data/processed', 'models', 'results/figures', 'notebooks', 'src']
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    print("✅ Folder structure created")

def save_model(model, scaler, model_name):
    """Save trained model and scaler"""
    joblib.dump(model, f'models/{model_name}_model.pkl')
    joblib.dump(scaler, f'models/{model_name}_scaler.pkl')
    print(f"✅ Saved {model_name} model and scaler")

def load_model(model_name):
    """Load trained model and scaler"""
    model = joblib.load(f'models/{model_name}_model.pkl')
    scaler = joblib.load(f'models/{model_name}_scaler.pkl')
    return model, scaler

def plot_confusion_matrix(y_true, y_pred, title, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved confusion matrix to {save_path}")

def plot_feature_importance(feature_names, importance_values, title, save_path, top_n=10):
    """Plot and save feature importance"""
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_values
    }).sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('Importance')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved feature importance plot to {save_path}")
    return importance_df

def print_metrics_report(y_true, y_pred, model_name, class_names):
    """Print detailed classification metrics"""
    print(f"\n{'='*60}")
    print(f"{model_name} - CLASSIFICATION REPORT")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Calculate additional metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"\n{'='*60}")
    print(f"SUMMARY METRICS")
    print(f"{'='*60}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }