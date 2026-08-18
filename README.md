# ML Classification Dashboard

## Problem Statement

Build an interactive machine learning classification dashboard that trains and evaluates
multiple classification models on a given dataset, displaying evaluation metrics,
confusion matrices, and model comparisons through a Streamlit web application.

## Dataset Description

**Dataset:** Wine Quality Classification (UCI Machine Learning Repository)

- **Source:** [UCI ML Repository - Wine Quality Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/)
- **Type:** Multi-class Classification
- **Total Instances:** 6,497 samples
  - Red Wine: 1,599 samples
  - White Wine: 4,898 samples
- **Features:** 12 total
  - **Physicochemical Features (11):** 
    - Fixed acidity, Volatile acidity, Citric acid, Residual sugar, Chlorides
    - Free sulfur dioxide, Total sulfur dioxide, Density, pH, Sulphates, Alcohol
  - **Additional Feature (1):** Wine type (0 = Red, 1 = White)
- **Target Column:** `quality` (Integer scale: 3–9, representing wine quality rating)
- **Missing Values:** None
- **Data Split:** 80% training, 20% testing

> This dataset contains physicochemical (input) and sensory (output) variables of red and white variants of the Portuguese "Vinho Verde" wine.

## GitHub Repository Link

[https://github.com/otakushreyas/ml-classification-dashboard/tree/feature/your-feature](https://github.com/otakushreyas/ml-classification-dashboard/tree/feature/your-feature)

## Models Used

The following 5 classification models are implemented and evaluated:

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.5408 | 0.7229 | 0.5415 | 0.5408 | 0.5132 | 0.2707 |
| Decision Tree | 0.5600 | 0.7129 | 0.5517 | 0.5600 | 0.5522 | 0.3336 |
| K-Nearest Neighbors | 0.6531 | 0.8359 | 0.6452 | 0.6531 | 0.6457 | 0.4739 |
| Naive Bayes (Gaussian) | 0.3215 | 0.5966 | 0.4207 | 0.3215 | 0.3621 | 0.0996 |
| Random Forest (Ensemble) | 0.6800 | 0.8535 | 0.6849 | 0.6800 | 0.6660 | 0.5065 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Moderate performance with balanced metrics. Good AUC (0.7229) suggests decent discriminative ability but lower accuracy (0.5408) indicates room for improvement. |
| Decision Tree | Slightly better accuracy than Logistic Regression (0.5600) but lower AUC. Shows signs of potential overfitting; tuning hyperparameters could help improve generalization. |
| K-Nearest Neighbors | **Best performer** with highest accuracy (0.6531), best F1 score (0.6457), and excellent AUC (0.8359). Demonstrates strong and consistent performance across all evaluation metrics. |
| Naive Bayes (Gaussian) | Poorest performance across all metrics. Very low accuracy (0.3215) and MCC (0.0996) suggest that Naive Bayes assumptions are not well-suited for this dataset's feature distributions. |
| Random Forest (Ensemble) | Strong performer with 0.68 accuracy and excellent AUC (0.8535). Higher MCC (0.5065) compared to other models indicates better overall classification quality. Balanced precision and recall suggest good generalization. |
| **Overall Winner** | **K-Nearest Neighbors** ⭐ – Highest accuracy (0.6531) and F1 score (0.6457) make it the most reliable model for this classification task, though Random Forest is a close second with better MCC. |

## How to Run Locally

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd <project-folder>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py
```

## Live App

🔗 **Streamlit App:** *(Add your deployed Streamlit Cloud URL here)*

## Project Structure

```
project-folder/
│── app.py                  # Streamlit web application
│── requirements.txt        # Python dependencies
│── README.md               # This file
│── test_data.csv           # Test data for experiments
│── model/
│   └── train_models.py     # Model training & evaluation script
```

## Tech Stack

- **Python 3.x**
- **Streamlit** – Interactive web UI
- **scikit-learn** – ML models & metrics
- **pandas / numpy** – Data manipulation
- **matplotlib / seaborn** – Visualisation
