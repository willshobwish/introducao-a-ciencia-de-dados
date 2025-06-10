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

def objective(trial:optuna.Trial):
    transformers = []
    scaler_options = {
    "standard": StandardScaler(),
    "minmax": MinMaxScaler(),
    "robust": RobustScaler(),
    "ordinal": OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
    "none":'passthrough'
    }

    model = RandomForestClassifier()

    for col in X.columns:
        scaler_choice = trial.suggest_categorical(f"scaler_{col}", list(scaler_options.keys()))
        transformers.append((f"{col}_scaler", scaler_options[scaler_choice], [col]))

    col_transformer = ColumnTransformer(transformers)
    steps = [("scaler", col_transformer)]

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
                                storage="sqlite:///random_forest_scaler.db",
                                load_if_exists=True,
                                study_name="no-name-7d86f460-e62a-450d-ad4f-d09a17fa12dc")
    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"Trials completos até agora: {completed_trials}")

    remaining_trials = MAX_TRIALS - completed_trials

    if remaining_trials > 0:
        print(f"Executando {remaining_trials} trials restantes...")
        study.optimize(objective, n_trials=remaining_trials)
    else:
        print("Número máximo de trials já atingido. Nenhum novo trial será executado.")
