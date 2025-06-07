from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import optuna

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
        "eval_metric": "mlogloss"
    }

    model = XGBClassifier(**params)

    dropout_scores = []
    for _ in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        f1_scores = f1_score(y_test, preds, average=None)
        dropout_index = le.transform(["Dropout"])[0]
        dropout_scores.append(f1_scores[dropout_index])
    return np.mean(dropout_scores)

if __name__ == "__main__":
    df = pd.read_csv(r'predict_students_dropout_and_academic_success.csv')
    df = df[df["Target"] != "Enrolled"]
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    le = LabelEncoder() 
    y = le.fit_transform(y)
    
    study = optuna.create_study(direction="maximize",
                                storage="sqlite:///xgboost.db",
                                load_if_exists=True,
                                study_name="xgboost")
    study.optimize(objective, n_trials=1000)