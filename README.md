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
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.994615 | 0.998925 | 0.994611 | 0.994615 | 0.994613 | 0.985478 |
| Decision Tree | 0.990000 | 0.985199 | 0.990045 | 0.990000 | 0.990016 | 0.973158 |
| K-Nearest Neighbors | 0.993077 | 0.998318 | 0.993113 | 0.993077 | 0.993088 | 0.981423 |
| Naive Bayes (Gaussian) | 0.976923 | 0.994088 | 0.977543 | 0.976923 | 0.977087 | 0.939106 |
| Random Forest (Ensemble) | 0.996923 | 0.999949 | 0.996923 | 0.996923 | 0.996923 | 0.991709 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Very strong performance with accuracy of 0.9946 and AUC of 0.9989. It is highly competitive and delivers consistently balanced precision, recall and F1. |
| Decision Tree | Strong classification performance with 0.99 accuracy and good MCC of 0.9732. It is slightly behind the top models but still highly reliable. |
| K-Nearest Neighbors | Excellent model with 0.9931 accuracy and AUC of 0.9983. It performs very well across all metrics and remains close to the best model. |
| Naive Bayes (Gaussian) | Performs well but trails the other models, with accuracy of 0.9769 and MCC of 0.9391. It remains competitive but is less precise than the top performers. |
| Random Forest (Ensemble) | **Best performer** with the highest accuracy (0.9969), near-perfect AUC (0.999949), and the strongest MCC (0.991709). It offers the most robust overall classification quality. |
| **Overall Winner** | **Random Forest (Ensemble)** ⭐ – Highest accuracy and AUC, along with the best MCC and nearly perfect F1, make it the strongest model for this classification task. |

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

🔗 **Streamlit App:** [https://ml-classification-dashboard-vnz5wycywszfczdv6bwrbp.streamlit.app/](https://ml-classification-dashboard-vnz5wycywszfczdv6bwrbp.streamlit.app/)

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
