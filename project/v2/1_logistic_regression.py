import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, confusion_matrix, ConfusionMatrixDisplay,classification_report
import optuna
from random import randint
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
from optuna.trial import Trial

def objective(trial:Trial):
    random_state_train_test_split = randint(0,1000)
    trial.set_user_attr("random_state_train_test_split", random_state_train_test_split)
    
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=random_state_train_test_split
    )

    C = trial.suggest_float("C", 1e-3, 1e3, log=True)
    penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
    solver = "liblinear" if penalty == "l1" else trial.suggest_categorical("solver", ["lbfgs", "saga"])
    max_iter_clf = trial.suggest_int("max_iter_clf",100,10000)

    random_state_clf = randint(0,1000)
    random_state_cv = randint(0,1000)
    trial.set_user_attr("random_state_clf", random_state_clf)
    trial.set_user_attr("random_state_cv", random_state_cv)

    pipe = Pipeline([
        ("clf", LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=max_iter_clf, random_state=random_state_clf)),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state_cv)

    recall_scores = []  

    for train_idx, valid_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_tr, y_val = y_train[train_idx], y_train[valid_idx]

        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_val)
        recall_scores.append(recall_score(y_val, preds, average='weighted'))

        if np.mean(recall_scores) == 0:
            # Raise TrialPruned or simply RuntimeError to retry this trial
            raise optuna.TrialPruned()  # tells Optuna to skip this trial and try another
        else:
            return np.mean(recall_scores)
        
if __name__ == "__main__":
    df = pd.read_csv(r'predict_students_dropout_and_academic_success.csv')

    X = df.iloc[:, :-1]   # all rows, all columns except the last
    y = df.iloc[:,  -1]   # all rows, just the last column

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    # Run Optuna
    study = optuna.create_study(direction="maximize",study_name="no-name-e61ce508-2186-43ad-828f-bb8f0a6b8cc9",storage="sqlite:///logistic_regression_w.db",load_if_exists=True)
    study.optimize(objective, n_trials=606)

    # Results
    print("Best trial:")
    print(f"Recall: {study.best_value}")
    print("Params:")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")
        
    C = study.best_params["C"]
    penalty = study.best_params["penalty"]
    solver = study.best_params["solver"]
    max_iter_clf = study.best_params["max_iter_clf"]

    pipe = Pipeline([
        ("clf", LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=max_iter_clf)),
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    
    preds_decoded = le.inverse_transform(preds)
    y_test_decoded = le.inverse_transform(y_test)
    report = classification_report(y_test_decoded, preds_decoded)
    print(report)
    cm = confusion_matrix(y_test_decoded, preds_decoded,labels=le.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=le.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()
