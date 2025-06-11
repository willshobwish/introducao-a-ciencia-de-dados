from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import optuna
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from multiprocessing import Pool
import functools

def evaluate_model(args, trial, model_params, steps):
    X, y, le = args
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    
    # Create a new pipeline for this iteration
    pipe = Pipeline(steps)
    
    # Set model parameters correctly using the step name
    pipe.set_params(**{'classifier__' + k: v for k, v in model_params.items()})
    
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    f1_score_per_class = f1_score(y_test, preds, average=None)
    dropout_index = le.transform(["Dropout"])[0]
    return f1_score_per_class[dropout_index]

def objective(trial: optuna.Trial):
    model_params = {
        'n_estimators': trial.suggest_int("n_estimators", 30, 500),
        'max_depth': trial.suggest_int("max_depth", 3, 60),
        'min_samples_split': trial.suggest_int("min_samples_split", 2, 40),
        'min_samples_leaf': trial.suggest_int("min_samples_leaf", 1, 40),
        'max_features': trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        'bootstrap': trial.suggest_categorical("bootstrap", [True, False])
    }
        
    model = RandomForestClassifier()
    pca_usage = trial.suggest_categorical("pca_usage", [None, "pca"])
    rfe_usage = trial.suggest_categorical("rfe_usage", [None, "rfe"])

    transformers = []
    scaler_options = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
        "ordinal": OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
        "none": 'passthrough'
    }
    
    for col in X.columns:
        scaler_choice = trial.suggest_categorical(f"scaler_{col}", list(scaler_options.keys()))
        transformers.append((f"{col}_scaler", scaler_options[scaler_choice], [col]))

    col_transformer = ColumnTransformer(transformers)
    steps = [("preprocessor", col_transformer)]

    if pca_usage:
        n_components = trial.suggest_int("pca__n_components", 2, X.shape[1])
        svd_solver = trial.suggest_categorical("pca__svd_solver", ["auto", "full", "arpack", "randomized"])
        whiten = trial.suggest_categorical("pca__whiten", [True, False])

        if svd_solver == "arpack":
            n_components = min(n_components, X.shape[1] - 1)

        pca = PCA(
            n_components=n_components,
            svd_solver=svd_solver,
            whiten=whiten
        )
        steps.append(("pca", pca))

    if rfe_usage:
        if pca_usage:
            n_features = trial.suggest_int("rfe__n_features_to_select", 1, n_components)
        else:
            n_features = trial.suggest_int("rfe__n_features_to_select", 1, X.shape[1])

        rfe_step = trial.suggest_float("rfe__step", 0.1, 1.0)
        rfe = RFE(
            estimator=RandomForestClassifier(n_estimators=10),  # Lightweight estimator for RFE
            n_features_to_select=n_features,
            step=rfe_step
        )
        steps.append(("rfe", rfe))
    
    # Final model step must be named 'classifier' for set_params to work
    steps.append(("classifier", model))
    
    # Prepare arguments for parallel processing
    args_list = [(X, y, le) for _ in range(10)]
    
    # Create a partial function with the trial and steps fixed
    eval_func = functools.partial(evaluate_model, trial=trial, model_params=model_params, steps=steps)
    
    # Use multiprocessing with limited workers
    with Pool(processes=min(4, os.cpu_count())) as pool:  # Limit workers to prevent memory issues
        f1_scores = pool.map(eval_func, args_list)
    
    statistic, pvalue = wilcoxon(x=f1_scores)

    trial.set_user_attr("var", np.var(f1_scores))
    trial.set_user_attr("std", np.std(f1_scores))
    trial.set_user_attr("wilcoxon_statistic", statistic)
    trial.set_user_attr("wilcoxon_pvalue", pvalue)
    return np.mean(f1_scores)

if __name__ == "__main__":
    import os
    df = pd.read_csv('predict_students_dropout_and_academic_success.csv')
    df = df[df["Target"] != "Enrolled"]
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    le = LabelEncoder() 
    y = le.fit_transform(y)

    MAX_TRIALS = 1000
    
    # Configure SQLite storage with optimized settings
    storage = optuna.storages.RDBStorage(
        url="sqlite:///scaler_pca_rfe.db",
        engine_kwargs={
            "pool_size": 10,
            "max_overflow": 0,
            "pool_pre_ping": True
        }
    )
    
    study = optuna.create_study(
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        study_name="optimized_study"
    )
    
    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"Completed trials: {completed_trials}")

    remaining_trials = MAX_TRIALS - completed_trials

    if remaining_trials > 0:
        print(f"Running {remaining_trials} remaining trials...")
        study.optimize(objective, n_trials=remaining_trials, n_jobs=1)  # n_jobs=1 to avoid nested parallelism
    else:
        print("Maximum number of trials already reached.")