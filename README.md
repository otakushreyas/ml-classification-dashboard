# ML Classification Dashboard

## Problem Statement

Build an interactive machine learning classification dashboard that trains and evaluates
multiple classification models on a given dataset, displaying evaluation metrics,
confusion matrices, and model comparisons through a Streamlit web application.

## Dataset Description

**Dataset:** *(Update with your chosen dataset name and source)*

- **Source:** Kaggle / UCI Machine Learning Repository
- **Type:** Classification (binary / multi-class)
- **Features:** *(update count)*
- **Instances:** *(update count)*
- **Target Column:** *(update column name)*

> Upload your dataset CSV in the Streamlit app and select the target column to get started.

## GitHub Repository Link

*(Add your GitHub repo URL here)*

## Models Used

The following 5 classification models are implemented and evaluated:

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | — | — | — | — | — | — |
| Decision Tree | — | — | — | — | — | — |
| K-Nearest Neighbors | — | — | — | — | — | — |
| Naive Bayes (Gaussian) | — | — | — | — | — | — |
| Random Forest (Ensemble) | — | — | — | — | — | — |

> *(Run the app and fill in these values from the All-Model Comparison table)*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | *(Add observation)* |
| Decision Tree | *(Add observation)* |
| K-Nearest Neighbors | *(Add observation)* |
| Naive Bayes (Gaussian) | *(Add observation)* |
| Random Forest (Ensemble) | *(Add observation)* |
| **Overall Winner** | *(Add winner and reasoning)* |

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
