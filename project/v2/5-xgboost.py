import pandas as pd
import optuna
from random import randint
import xgboost as xgb
from sklearn.metrics import recall_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Objective function for Optuna
def objective(trial):
    train_test_split_random_state = randint(0, 1000)
    trial.set_user_attr("train_test_split_random_state", train_test_split_random_state)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=train_test_split_random_state
    )

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        'max_depth': trial.suggest_int("max_depth", 3, 100),
        'learning_rate': trial.suggest_float("learning_rate", 0.001, 0.5),
        'subsample': trial.suggest_float("subsample", 0.1, 1.0),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.1, 1.0),
        'tree_method': 'hist',     # Fast histogram algorithm
        # 'device': 'cuda',          # Use GPU
        'eval_metric': 'logloss',
        'objective': 'multi:softmax',  # Use 'binary:logistic' for binary classification
        'num_class': len(set(y)),      # Required for multi-class
    }

    bst = xgb.train(params, dtrain, num_boost_round=trial.suggest_int("n_estimators", 10, 2000))
    preds = bst.predict(dtest)

    return recall_score(y_test, preds, average='weighted')


if __name__ == "__main__":
    df = pd.read_csv(r'predict_students_dropout_and_academic_success.csv')

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    le = LabelEncoder()
    y = le.fit_transform(y)

    study = optuna.create_study(
        direction="maximize",
        study_name="XGB_DMatrix",
        storage="sqlite:///xgboost_dmatrix.db",
        load_if_exists=True
    )

    study.optimize(objective, n_trials=3000)

    # Train best model on full data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )
    
    d_train = xgb.DMatrix(X_train, label=y_train)
    d_test = xgb.DMatrix(X_test, label=y_test)

    best_params = study.best_params
    best_params.update({
        'tree_method': 'hist',
        'device': 'cuda',
        'eval_metric': 'logloss',
        'objective': 'multi:softmax',
        'num_class': len(set(y)),
    })
    booster = xgb.train(best_params, d_train, num_boost_round=best_params["n_estimators"])
    
    
    preds = booster.predict(d_test)
    
    # recall_score(y_test, preds, average='weighted')
    print(classification_report(y_test, preds))