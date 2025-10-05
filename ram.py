# Football match outcome prediction - Complete notebook code

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, mean_squared_error, precision_score, recall_score
import joblib

# 1. Load data
DATA_PATH = "football_synthetic_5000.csv"  # change if needed
df = pd.read_csv(DATA_PATH, parse_dates=["date"])

print("Rows:", len(df))
print(df.head())

# 2. Quick EDA
print(df['result'].value_counts(normalize=True))
plt.figure(figsize=(6,4))
sns.countplot(x='result', data=df)
plt.title("Result distribution (H=Home win, D=Draw, A=Away win)")
plt.show()

# 3. Feature engineering
df['rank_diff'] = df['home_rank'] - df['away_rank']
df['form_diff'] = df['home_form'] - df['away_form']
df['exp_goals_diff'] = df['home_expected_goals'] - df['away_expected_goals']

# Encode teams
le_home = LabelEncoder()
le_away = LabelEncoder()
df['home_enc'] = le_home.fit_transform(df['home_team'])
df['away_enc'] = le_away.fit_transform(df['away_team'])

# Target encoding: H=0, D=1, A=2
target_map = {'H':0,'D':1,'A':2}
df['target'] = df['result'].map(target_map)

features = ['home_enc','away_enc','rank_diff','form_diff','exp_goals_diff']
X = df[features]
y = df['target']

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# 5. Scale numerical features
scaler = StandardScaler()
num_cols = ['rank_diff','form_diff','exp_goals_diff']
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# 6. Train multiple models
models = {
    "LogisticRegression": LogisticRegression(multi_class='multinomial', max_iter=200),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# Try XGBoost optionally
try:
    import xgboost as xgb
    models["XGBoost"] = xgb.XGBClassifier(eval_metric='mlogloss', random_state=42)
except:
    print("XGBoost not installed")

results = {}

for name, model in models.items():
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    start_time = time.time()
    pred = model.predict(X_test)
    test_time = time.time() - start_time

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='weighted')
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    precision = precision_score(y_test, pred, average='weighted')
    recall = recall_score(y_test, pred, average='weighted')

    results[name] = {
        "Model": model,
        "Accuracy": acc,
        "F1-Score": f1,
        "RMSE": rmse,
        "Precision": precision,
        "Recall": recall,
        "Train Time": train_time,
        "Test Time": test_time
    }

    print(f"\n{name} Results:")
    print(f"Accuracy: {acc:.3f}, F1: {f1:.3f}, RMSE: {rmse:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
    print(classification_report(y_test, pred, target_names=['HomeWin','Draw','AwayWin']))

# 7. Compare models
comparison_df = pd.DataFrame(results).T.sort_values(by="Accuracy", ascending=False)
print("\nModel Comparison:\n", comparison_df[["Accuracy","F1-Score","RMSE","Precision","Recall","Train Time","Test Time"]])

# 8. Choose best model
best_model_name = comparison_df.index[0]
best_model = results[best_model_name]["Model"]
print("\nBest Model:", best_model_name)

# 9. Confusion matrix
best_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=['HomeWin','Draw','AwayWin'], yticklabels=['HomeWin','Draw','AwayWin'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - {best_model_name}")
plt.show()

# 10. Save model and encoders
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_model.joblib")
joblib.dump(scaler, "models/scaler.joblib")
joblib.dump(le_home, "models/le_home.joblib")
joblib.dump(le_away, "models/le_away.joblib")
print("Saved model and encoders to ./models/")

# 11. Prediction function
def predict_match(home_team, away_team, home_rank, away_rank, home_form, away_form, model=best_model):
    row = {}
    try:
        row['home_enc'] = int(le_home.transform([home_team])[0])
    except:
        row['home_enc'] = -1
    try:
        row['away_enc'] = int(le_away.transform([away_team])[0])
    except:
        row['away_enc'] = -1
    row['rank_diff'] = home_rank - away_rank
    row['form_diff'] = home_form - away_form
    h_strength = 50 + (10 - row['home_enc']%10)
    a_strength = 50 + (10 - row['away_enc']%10)
    row['exp_goals_diff'] = ( (h_strength/(h_strength+a_strength)) * 2.2 + home_form*0.6 ) - ( (a_strength/(h_strength+a_strength)) * 1.4 + away_form*0.4 )
    Xr = pd.DataFrame([row])
    Xr[num_cols] = scaler.transform(Xr[num_cols])
    pred = model.predict(Xr[features])
    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(Xr[features])[0]
    label_map = {0:'HomeWin',1:'Draw',2:'AwayWin'}
    return label_map[int(pred[0])], prob

# 12. Example prediction
home_team = "ManCity"
away_team = "Arsenal"
home_rank = 1
away_rank = 5
home_form = 0.85
away_form = 0.60

result, probabilities = predict_match(home_team, away_team, home_rank, away_rank, home_form, away_form)
print("\nExample Match Prediction:")
print("Predicted Result:", result)
print("Probabilities [HomeWin, Draw, AwayWin]:", probabilities)
