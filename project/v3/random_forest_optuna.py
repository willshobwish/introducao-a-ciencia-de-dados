import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import optuna
from scipy.stats import wilcoxon
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

def objective(trial):
    # Sample hyperparameters from Optuna
    n_estimators = trial.suggest_int("n_estimators", 30, 500)
    max_depth = trial.suggest_int("max_depth", 3, 60)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 40)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 40)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])
    bootstrap = trial.suggest_categorical("bootstrap", [True, False])

    # Split data inside objective for a single fold (stratified)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap
    )
    
    model.fit(X_train, y_train)

    f1_scores = []

    for iteration in range(0, 10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

        model.fit(X_train,y_train)
        preds = model.predict(X_test)
        f1_score_per_class = f1_score(y_test, preds, average=None)
        dropout_index = le.transform(["Dropout"])[0]
        dropout_f1_score = f1_score_per_class[dropout_index]
        f1_scores.append(dropout_f1_score)
    
    statistic, pvalue = wilcoxon(x=f1_scores)

    trial.set_user_attr("var", np.var(f1_scores))
    trial.set_user_attr("std", np.std(f1_scores))
    trial.set_user_attr("wilcoxon_statistic", statistic)
    trial.set_user_attr("wilcoxon_pvalue", pvalue)
    return np.mean(f1_scores)

# Run Optuna optimization
if __name__ == "__main__":
    df = pd.read_csv(r'predict_students_dropout_and_academic_success.csv')
    df = df[df["Target"] != "Enrolled"]
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    le = LabelEncoder() 
    y = le.fit_transform(y)

    MAX_TRIALS = 1000

    study = optuna.create_study(direction="maximize",
                                storage="sqlite:///random_forest.db",
                                load_if_exists=True,
                                study_name="random_forest_optuna")
    study.optimize(objective, 
                   n_trials=1000,
                   callbacks=[MaxTrialsCallback(n_trials=1000, states=(TrialState.COMPLETE,))])


