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
    n_estimators = trial.suggest_int("n_estimators", 30, 500)
    max_depth = trial.suggest_int("max_depth", 3, 60)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 40)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 40)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])
    bootstrap = trial.suggest_categorical("bootstrap", [True, False])

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap
    )

    steps = []

    n_features = trial.suggest_int(name="rfe__n_features_to_select", low=1, high=X.shape[1])

    # Step: number or fraction of features to remove per iteration
    rfe_step = trial.suggest_float("rfe__step", 0.1, 1.0)

    rfe = RFE(estimator=model,
            n_features_to_select=n_features,
            step=rfe_step)

    steps.append(("rfe", rfe))
    
    # Modelo final
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
                                storage="sqlite:///rfe.db",
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
