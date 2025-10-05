# Football Match Outcome Prediction

This project predicts the outcome of football matches (Home Win, Draw, Away Win) using machine learning models. It demonstrates data preprocessing, model training, evaluation, and prediction.

## Dataset

- The dataset consists of **5000 synthetic football match records**.  
- Features include:  
  - `home_team`, `away_team`  
  - `home_rank`, `away_rank`  
  - `home_form`, `away_form`  
  - `home_expected_goals`, `away_expected_goals`  
- Target variable: `result` (H = Home Win, D = Draw, A = Away Win)

## Models Trained

- Logistic Regression  
- Random Forest Classifier  
- Decision Tree Classifier  
- K-Nearest Neighbors (KNN)  
- XGBoost Classifier (optional, if installed)  

## Evaluation Metrics

- Accuracy  
- F1-score  
- RMSE  
- Precision & Recall  
- Confusion Matrix  

## Model Comparison

| Model | Accuracy | F1-Score | RMSE | Precision | Recall |  
|-------|----------|----------|------|-----------|--------|  
| Logistic Regression | 0.514 | 0.371 | 1.068 | 0.371 | 0.514 |  
| Random Forest | 0.471 | 0.413 | 1.037 | 0.408 | 0.471 |  
| Decision Tree | 0.392 | 0.396 | 1.109 | 0.401 | 0.392 |  
| KNN | 0.472 | 0.420 | 1.058 | 0.424 | 0.472 |  
| XGBoost | 0.448 | 0.411 | 1.066 | 0.398 | 0.448 |  

**Best Model:** Logistic Regression

## Usage

1. Clone the repository:  
   ```bash
   git clone <your-repo-link>
   cd <repo-folder>
Install dependencies:

pip install -r requirements.txt
Place your CSV dataset in the project folder or link it from Google Drive (update DATA_PATH in the notebook).

Run the notebook or Python script:

python ram.py
Make predictions using the predict_match function:

from ram import predict_match

result, probabilities = predict_match(
    home_team="ManCity",
    away_team="Arsenal",
    home_rank=1,
    away_rank=5,
    home_form=0.85,
    away_form=0.60
)

print(result)
print(probabilities)
Files
ram.py – Python script with model training, evaluation, and prediction

football_synthetic_5000.csv – Dataset

models/ – Saved model and preprocessing objects (scaler, encoders)

Football_Match_Prediction.ipynb – Jupyter Notebook

Conclusion
Logistic Regression performed best overall.

Draws are harder to predict due to class imbalance.

Future work: Improve prediction with more data, additional features, and hyperparameter tuning.


If you want, I can also make a **ready-to-use `requirements.txt`** file for this project so anyone can insta
