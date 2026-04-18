# 🧠 Mental Health Screening System

## 📌 Project Overview

This project develops a **machine learning-based mental health screening system** for students that predicts:
1. **Stress Levels** (Low/Moderate/High) from lifestyle factors
2. **Depression Severity** (Minimal/Mild/Moderate/Moderately Severe/Severe) from PHQ-9 questionnaire responses

## 📊 Datasets Used

| Dataset | Samples | Features | Target |
|---------|---------|----------|--------|
| Student Stress Factors | 520 students | 5 lifestyle factors | Stress Level (1-5) |
| PHQ-9 Depression | 250 students | 9 questionnaire responses | Severity Level |

## 🤖 Models Used

### Stacking Ensemble Architecture

| Level | Model | Parameters |
|-------|-------|------------|
| **Base Model 1** | Random Forest | 100 trees, max_depth=10 |
| **Base Model 2** | XGBoost | 100 trees, max_depth=6, learning_rate=0.1 |
| **Meta-Learner** | Logistic Regression | L2 regularization, C=0.1 |

### Why Stacking Ensemble?
- Combines strengths of multiple algorithms
- Random Forest handles non-linear relationships well
- XGBoost excels at capturing complex patterns
- Logistic Regression as meta-learner prevents overfitting

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Stress Prediction** | **91.35%** | 91.50% | 91.35% | 91.38% |
| **Depression Prediction** | **100%** | 100% | 100% | 100% |

### Stress Model - Per Class Performance

| Stress Level | Precision | Recall | F1-Score |
|--------------|-----------|--------|----------|
| Low | 98% | 95% | 96% |
| Moderate | 85% | 92% | 88% |
| High | 89% | 86% | 87% |

## 🔑 Key Findings

- **Sleep quality** is the most important predictor of student stress
- **PHQ-9** is highly reliable for depression screening (100% accuracy)
- Stacking ensemble improved accuracy by 5% over individual models

## 💡 Why 100% Depression Accuracy?

PHQ-9 is a standardized clinical tool where severity is **mathematically calculated** from responses (total score 0-27 maps to fixed severity levels), making prediction deterministic.

## 🛠️ Technologies Used

- Python 3.11
- Pandas, NumPy (Data processing)
- Scikit-learn, XGBoost (Machine Learning)
- Matplotlib, Seaborn (Visualization)

---

**Built for early mental health screening in educational institutions**