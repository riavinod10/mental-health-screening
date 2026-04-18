"""
Train Stacking Ensemble Models for Stress and Depression Prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import save_model, plot_confusion_matrix, plot_feature_importance, print_metrics_report
from src.feature_engineering import engineer_stress_features, engineer_depression_features, scale_features

def train_stress_model():
    """
    Train stacking ensemble for stress prediction
    """
    
    print("\n" + "="*70)
    print("TRAINING STACKING ENSEMBLE - STRESS LEVEL PREDICTION")
    print("="*70)
    
    # Load processed data
    df = pd.read_csv('data/processed/stress_processed.csv')
    
    # Feature engineering
    df, feature_cols = engineer_stress_features(df)
    
    # Prepare X and y
    X = df[feature_cols]
    y = df['stress_category']  # 0=Low, 1=Moderate, 2=High
    
    print(f"\n📊 Dataset Info:")
    print(f"   Samples: {len(X)}")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Classes: {y.nunique()}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, feature_cols)
    
    # Define base models for stacking ensemble
    base_models = [
        ('random_forest', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )),
        ('xgboost', XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss'
        ))
    ]
    
    # Meta-learner
    meta_learner = LogisticRegression(
        C=0.1,
        penalty='l2',
        max_iter=1000,
        random_state=42
    )
    
    # Create stacking ensemble
    stacking_model = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=5,
        stack_method='predict_proba',
        passthrough=False,
        n_jobs=-1
    )
    
    # Train individual models for comparison
    print("\n📈 Training individual models...")
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"   {name}: {acc:.4f}")
    
    # Train stacking ensemble
    print("\n🎯 Training Stacking Ensemble...")
    stacking_model.fit(X_train_scaled, y_train)
    y_pred_stack = stacking_model.predict(X_test_scaled)
    stack_acc = accuracy_score(y_test, y_pred_stack)
    results['Stacking Ensemble'] = stack_acc
    print(f"   Stacking Ensemble: {stack_acc:.4f}")
    
    # Show improvement
    best_individual = max(results['Random Forest'], results['XGBoost'])
    improvement = (stack_acc - best_individual) * 100
    print(f"\n📈 Stacking improvement: +{improvement:.2f}% over best individual model")
    
    # Cross-validation
    print("\n🔄 Cross-validation (5-fold) on Stacking Ensemble...")
    cv_scores = cross_val_score(stacking_model, X_train_scaled, y_train, cv=5)
    print(f"   CV Scores: {cv_scores}")
    print(f"   Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Feature importance from Random Forest
    rf_model = stacking_model.named_estimators_['random_forest']
    feature_importance = plot_feature_importance(
        feature_cols,
        rf_model.feature_importances_,
        "Stress Prediction - Feature Importance",
        "results/figures/stress_feature_importance.png",
        top_n=10
    )
    
    # Confusion matrix
    plot_confusion_matrix(
        y_test, y_pred_stack,
        "Stress Level Prediction - Confusion Matrix",
        "results/figures/stress_confusion_matrix.png"
    )
    
    # Metrics report
    class_names = ['Low Stress', 'Moderate Stress', 'High Stress']
    metrics = print_metrics_report(y_test, y_pred_stack, "STRESS MODEL", class_names)
    
    # Save model and scaler
    save_model(stacking_model, scaler, 'stress')
    
    # Save results summary
    results_summary = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': list(results.values())
    })
    results_summary.to_csv('results/stress_model_comparison.csv', index=False)
    print("\n✅ Stress model training completed!")
    
    return stacking_model, scaler, metrics

def train_depression_model():
    """
    Train stacking ensemble for depression prediction
    """
    
    print("\n" + "="*70)
    print("TRAINING STACKING ENSEMBLE - DEPRESSION RISK PREDICTION")
    print("="*70)
    
    # Load processed data
    df = pd.read_csv('data/processed/depression_processed.csv')
    
    # Get feature columns (all _score columns)
    feature_cols = [col for col in df.columns if col.endswith('_score')]
    
    # Feature engineering
    df, feature_cols = engineer_depression_features(df, feature_cols)
    
    # Prepare X and y
    X = df[feature_cols]
    y = df['target']  # 0=Minimal, 1=Mild, 2=Moderate, 3=Moderately Severe, 4=Severe
    
    print(f"\n📊 Dataset Info:")
    print(f"   Samples: {len(X)}")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Classes: {y.nunique()}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, feature_cols)
    
    # Define base models for stacking ensemble
    base_models = [
        ('random_forest', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )),
        ('xgboost', XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss'
        ))
    ]
    
    # Meta-learner
    meta_learner = LogisticRegression(
        C=0.1,
        penalty='l2',
        max_iter=1000,
        random_state=42
    )
    
    # Create stacking ensemble
    stacking_model = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=5,
        stack_method='predict_proba',
        passthrough=False,
        n_jobs=-1
    )
    
    # Train individual models for comparison
    print("\n📈 Training individual models...")
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"   {name}: {acc:.4f}")
    
    # Train stacking ensemble
    print("\n🎯 Training Stacking Ensemble...")
    stacking_model.fit(X_train_scaled, y_train)
    y_pred_stack = stacking_model.predict(X_test_scaled)
    stack_acc = accuracy_score(y_test, y_pred_stack)
    results['Stacking Ensemble'] = stack_acc
    print(f"   Stacking Ensemble: {stack_acc:.4f}")
    
    # Show improvement
    best_individual = max(results['Random Forest'], results['XGBoost'])
    improvement = (stack_acc - best_individual) * 100
    print(f"\n📈 Stacking improvement: +{improvement:.2f}% over best individual model")
    
    # Cross-validation
    print("\n🔄 Cross-validation (5-fold) on Stacking Ensemble...")
    cv_scores = cross_val_score(stacking_model, X_train_scaled, y_train, cv=5)
    print(f"   CV Scores: {cv_scores}")
    print(f"   Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Feature importance from Random Forest
    rf_model = stacking_model.named_estimators_['random_forest']
    feature_importance = plot_feature_importance(
        feature_cols,
        rf_model.feature_importances_,
        "Depression Prediction - Feature Importance",
        "results/figures/depression_feature_importance.png",
        top_n=10
    )
    
    # Confusion matrix
    plot_confusion_matrix(
        y_test, y_pred_stack,
        "Depression Risk Prediction - Confusion Matrix",
        "results/figures/depression_confusion_matrix.png"
    )
    
    # Metrics report
    class_names = ['Minimal', 'Mild', 'Moderate', 'Moderately Severe', 'Severe']
    metrics = print_metrics_report(y_test, y_pred_stack, "DEPRESSION MODEL", class_names)
    
    # Save model and scaler
    save_model(stacking_model, scaler, 'depression')
    
    # Save results summary
    results_summary = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': list(results.values())
    })
    results_summary.to_csv('results/depression_model_comparison.csv', index=False)
    print("\n✅ Depression model training completed!")
    
    return stacking_model, scaler, metrics

if __name__ == "__main__":
    # Create folders first
    from src.utils import create_folder_structure
    create_folder_structure()
    
    # Train both models
    stress_model, stress_scaler, stress_metrics = train_stress_model()
    depression_model, depression_scaler, depression_metrics = train_depression_model()
    
    print("\n" + "="*70)
    print("🎉 ALL MODELS TRAINED SUCCESSFULLY!")
    print("="*70)
    print(f"\nStress Model Accuracy: {stress_metrics['accuracy']:.4f}")
    print(f"Depression Model Accuracy: {depression_metrics['accuracy']:.4f}")