import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import cupy as cp
import pandas as pd
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
import optuna 

def check_xgb_gpu(model):
    attrs = model.get_booster().attributes()
    gpu_used = False
    if 'gpu_id' in attrs:
        print(f"XGBoost GPU ID used: {attrs['gpu_id']}")
        gpu_used = True
    elif 'device' in attrs and 'gpu' in attrs['device']:
        print(f"XGBoost device: {attrs['device']}")
        gpu_used = True
    else:
        print("XGBoost GPU not used.")
    return gpu_used

def check_lgb_gpu(model):
    # LightGBM prints info if verbose is set, but we can inspect params
    params = model.get_params()
    if 'device' in params and params['device'] == 'gpu':
        print("LightGBM set to use GPU.")
        return True
    else:
        print("LightGBM GPU not used.")
        return False

def check_cat_gpu(model):
    params = model.get_params()
    if params.get('task_type', '').lower() == 'gpu':
        print("CatBoost set to use GPU.")
        return True
    else:
        print("CatBoost GPU not used.")
        return False

# Evaluation Function
def evaluate_model(model, X_test, y_test, name="Model"):
    y_pred = model.predict(X_test)

    # Only call .get() if it's a CuPy array
    if isinstance(y_pred, cp.ndarray):
        y_pred = y_pred.get()
    if isinstance(y_test, cp.ndarray):
        y_test = y_test.get()
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"\n=== {name} Metrics ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    return acc, prec, rec, f1

# 1. Optuna Study for XGBoost
def xgb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int("n_estimators", 10, 2000),
        'max_depth': trial.suggest_int("max_depth", 3, 100),
        'learning_rate': trial.suggest_float("learning_rate", 0.001, 0.5),
        'subsample': trial.suggest_float("subsample", 0.1, 1.0),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.1, 1.0),
        'tree_method': 'hist',  # Use GPU
        'device':'cuda',
        'eval_metric': 'logloss'
    }
    model = XGBClassifier(**params)
    model.fit(X_train_gpu, y_train_gpu)
    return accuracy_score(y_test, model.predict(X_test))

if __name__ == "__main__":

    df = pd.read_csv("predict_students_dropout_and_academic_success.csv",delimiter=";")
    le = LabelEncoder()
    df["Target"] = le.fit_transform(df["Target"])

    # Assume df has features and df_encoded["Target"] is the encoded label
    X = df.iloc[:,:-1]
    y = df["Target"]

    # Optional: split to test performance later
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    n_trials = 200
    max_trials_callback = n_trials * 4
    storage = "postgresql://myuser:mypassword@localhost:5432/mydatabase"

    X_train_gpu = cp.array(X_train)
    X_test_gpu = cp.array(X_test)
    y_train_gpu = cp.array(y_train)
    y_test_gpu = cp.array(y_test)

    xgb_study = optuna.create_study(direction="maximize", study_name="XGB",storage=storage, load_if_exists=True)
    xgb_study.optimize(xgb_objective, n_trials=n_trials,callbacks=[MaxTrialsCallback(max_trials_callback, states=(TrialState.COMPLETE,))])
    xgb_best_model = XGBClassifier(**xgb_study.best_params, tree_method='hist', device='cuda', eval_metric='logloss')
    xgb_best_model.fit(X_train_gpu, y_train_gpu)
    check_xgb_gpu(xgb_best_model)
    evaluate_model(xgb_best_model, X_test_gpu, y_test_gpu, "XGBoost")
