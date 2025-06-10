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

def objective(trial:optuna.Trial):
    model = RandomForestClassifier()

    steps = []

    n_components = trial.suggest_int("pca__n_components", 2, X.shape[1])
    svd_solver = trial.suggest_categorical("pca__svd_solver", ["auto", "full", "arpack", "randomized"])
    whiten = trial.suggest_categorical("pca__whiten", [True, False])

    # Optional: limit solvers for smaller n_components
    if svd_solver == "arpack":
        # arpack only works with n_components < n_features
        n_components = min(n_components, X.shape[1] - 1)

    pca = PCA(
        n_components=n_components,
        svd_solver=svd_solver,
        whiten=whiten
    )
    steps.append(("pca", pca))

    steps.append(("classifier", model))
    pipe = Pipeline(steps)

    f1_scores = []

    for iteration in range(0, 10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

        pipe.fit(X_train,y_train)
        preds = pipe.predict(X_test)
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

if __name__ == "__main__":
    df = pd.read_csv(r'predict_students_dropout_and_academic_success.csv')
    df = df[df["Target"] != "Enrolled"]
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    le = LabelEncoder() 
    y = le.fit_transform(y)

    MAX_TRIALS = 1000

    study = optuna.create_study(direction="maximize",
                                storage="sqlite:///pca.db",
                                load_if_exists=True,
                                study_name="no-name-4edeb741-7c80-4857-9a8a-47563ca4f188")
    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"Trials completos até agora: {completed_trials}")

    remaining_trials = MAX_TRIALS - completed_trials

    if remaining_trials > 0:
        print(f"Executando {remaining_trials} trials restantes...")
        study.optimize(objective, n_trials=remaining_trials)
    else:
        print("Número máximo de trials já atingido. Nenhum novo trial será executado.")
