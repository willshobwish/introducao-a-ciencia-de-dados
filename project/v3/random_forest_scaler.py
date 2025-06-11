from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OrdinalEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from scipy.stats import wilcoxon
import optuna
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from tqdm import tqdm
import warnings

# Suppress sklearn warnings
warnings.filterwarnings('ignore')

def evaluate_iteration(args):
    """Evaluate a single iteration with given parameters"""
    X, y, le, trial_params, iteration = args
    try:
        # Create fresh pipeline for each evaluation
        transformers = []
        scaler_options = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
            "ordinal": OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
            "none": 'passthrough'
        }
        
        for col in X.columns:
            scaler_choice = trial_params[f"scaler_{col}"]
            transformers.append((f"{col}_scaler", scaler_options[scaler_choice], [col]))
        
        col_transformer = ColumnTransformer(transformers)
        
        model = RandomForestClassifier(
            n_estimators=trial_params['n_estimators'],
            max_depth=trial_params['max_depth'],
            min_samples_split=trial_params['min_samples_split'],
            min_samples_leaf=trial_params['min_samples_leaf'],
            max_features=trial_params['max_features'],
            bootstrap=trial_params['bootstrap'],
            n_jobs=1  # Important for ProcessPool
        )
        
        pipe = Pipeline([
            ("scaler", col_transformer),
            ("classifier", model)
        ])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=iteration
        )
        
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        f1_score_per_class = f1_score(y_test, preds, average=None)
        dropout_index = le.transform(["Dropout"])[0]
        return f1_score_per_class[dropout_index]
    
    except Exception as e:
        print(f"Iteration {iteration} failed: {str(e)}")
        return None

def objective(trial: optuna.Trial):
    # Suggest hyperparameters
    trial_params = {
        'n_estimators': trial.suggest_int("n_estimators", 30, 500),
        'max_depth': trial.suggest_int("max_depth", 3, 60),
        'min_samples_split': trial.suggest_int("min_samples_split", 2, 40),
        'min_samples_leaf': trial.suggest_int("min_samples_leaf", 1, 40),
        'max_features': trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        'bootstrap': trial.suggest_categorical("bootstrap", [True, False])
    }
    
    # Add scaler choices for each column
    for col in X.columns:
        trial_params[f"scaler_{col}"] = trial.suggest_categorical(
            f"scaler_{col}", ["standard", "minmax", "robust", "ordinal", "none"]
        )
    
    # Parallel evaluation
    f1_scores = []
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [
            executor.submit(
                evaluate_iteration, 
                (X, y, le, trial_params, i)
            ) 
            for i in range(n_iterations)
        ]
        
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                f1_scores.append(result)
    
    # Calculate statistics
    if len(f1_scores) < 5:  # Minimum threshold for meaningful statistics
        raise optuna.TrialPruned()
    
    statistic, pvalue = wilcoxon(x=f1_scores)
    
    trial.set_user_attr("var", np.var(f1_scores))
    trial.set_user_attr("std", np.std(f1_scores))
    trial.set_user_attr("wilcoxon_statistic", statistic)
    trial.set_user_attr("wilcoxon_pvalue", pvalue)
    return np.mean(f1_scores)

if __name__ == "__main__":
    # Load and prepare data
    df = pd.read_csv(r'predict_students_dropout_and_academic_success.csv')
    df = df[df["Target"] != "Enrolled"]
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # Optuna study setup
    MAX_TRIALS = 1000
    study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///random_forest_scaler.db",
        load_if_exists=True,
        study_name="no-name-7d86f460-e62a-450d-ad4f-d09a17fa12dc"
    )
    
    # Check completed trials
    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"Completed trials: {completed_trials}")
    
    remaining_trials = MAX_TRIALS - completed_trials
    if remaining_trials > 0:
        print(f"Running {remaining_trials} remaining trials...")
        study.optimize(objective, n_trials=remaining_trials, n_jobs=1)  # n_jobs=1 to avoid nested parallelism
    else:
        print("Maximum number of trials already reached.")