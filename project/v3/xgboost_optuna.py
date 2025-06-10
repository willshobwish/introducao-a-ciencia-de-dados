from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OrdinalEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from scipy.stats import wilcoxon
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import optuna

# Load dataset
df = pd.read_csv('predict_students_dropout_and_academic_success.csv')
df = df[df["Target"] != "Enrolled"]

X = df.drop(columns="Target")
y = df["Target"]

le = LabelEncoder()
y = le.fit_transform(y)

def objective(trial: optuna.Trial):
    # XGBoost hyperparameters
    model = XGBClassifier(
        n_estimators=trial.suggest_int("n_estimators", 50, 500),
        max_depth=trial.suggest_int("max_depth", 3, high=50),
        learning_rate=trial.suggest_float("learning_rate", 0.001, 0.5, log=True),
        subsample=trial.suggest_float("subsample", 0.3, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
        gamma=trial.suggest_float("gamma", 0, 5),
        reg_alpha=trial.suggest_float("reg_alpha", 0, 5),
        reg_lambda=trial.suggest_float("reg_lambda", 0, 5),
        use_label_encoder=False,
        eval_metric="mlogloss",
        verbosity=0
    )

    pca_usage = trial.suggest_categorical("pca_usage", [None, "pca"])
    rfe_usage = trial.suggest_categorical("rfe_usage", [None, "rfe"])

    # Feature transformers
    scaler_options = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
        "ordinal": OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
        "none": 'passthrough'
    }

    transformers = []
    for col in X.columns:
        scaler_choice = trial.suggest_categorical(f"scaler_{col}", list(scaler_options.keys()))
        transformers.append((f"{col}_scaler", scaler_options[scaler_choice], [col]))

    col_transformer = ColumnTransformer(transformers)
    steps = [("scaler", col_transformer)]

    # Optional PCA
    if pca_usage:
        n_components = trial.suggest_int("pca__n_components", 2, X.shape[1])
        svd_solver = trial.suggest_categorical("pca__svd_solver", ["auto", "full", "arpack", "randomized"])
        whiten = trial.suggest_categorical("pca__whiten", [True, False])
        if svd_solver == "arpack":
            n_components = min(n_components, X.shape[1] - 1)
        steps.append(("pca", PCA(n_components=n_components, svd_solver=svd_solver, whiten=whiten)))

    # Optional RFE
    if rfe_usage:
        if pca_usage:
            n_features = trial.suggest_int("rfe__n_features_to_select", 1, n_components)
        else:
            n_features = trial.suggest_int("rfe__n_features_to_select", 1, X.shape[1])
        rfe_step = trial.suggest_float("rfe__step", 0.1, 1.0)
        steps.append(("rfe", RFE(estimator=model, n_features_to_select=n_features, step=rfe_step)))

    # Final model
    steps.append(("classifier", model))
    pipe = Pipeline(steps)

    f1_scores = []

    for _ in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        f1_per_class = f1_score(y_test, preds, average=None)
        dropout_index = le.transform(["Dropout"])[0]
        f1_scores.append(f1_per_class[dropout_index])

    # Wilcoxon test
    statistic, pvalue = wilcoxon(f1_scores)

    trial.set_user_attr("var", np.var(f1_scores))
    trial.set_user_attr("std", np.std(f1_scores))
    trial.set_user_attr("wilcoxon_statistic", statistic)
    trial.set_user_attr("wilcoxon_pvalue", pvalue)

    return np.mean(f1_scores)

# Run Optuna
if __name__ == "__main__":
    MAX_TRIALS = 1000
    study = optuna.create_study(direction="maximize", study_name="xgboost_pipeline", storage="sqlite:///xgboost_pipeline.db", load_if_exists=True)
    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"Trials completos até agora: {completed_trials}")

    remaining_trials = MAX_TRIALS - completed_trials

    if remaining_trials > 0:
        print(f"Executando {remaining_trials} trials restantes...")
        study.optimize(objective, n_trials=remaining_trials)
    else:
        print("Número máximo de trials já atingido. Nenhum novo trial será executado.")
