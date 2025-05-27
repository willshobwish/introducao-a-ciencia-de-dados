import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, confusion_matrix, ConfusionMatrixDisplay,classification_report
import optuna
from random import randint
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt

def objective(trial):
    random_state_train_test_split = randint(0,1000)
    trial.set_user_attr("random_state_train_test_split", random_state_train_test_split)
    
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=random_state_train_test_split
    )
    transformers = []
    scaler_options = {
    "standard": StandardScaler(),
    "minmax": MinMaxScaler(),
    "robust": RobustScaler(),
    "ordinal": OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
    "none":'passthrough'
    }
    
    for col in X.columns:
        scaler_choice = trial.suggest_categorical(f"scaler_{col}", list(scaler_options.keys()))
        transformers.append((f"{col}_scaler", scaler_options[scaler_choice], [col]))

    col_transformer = ColumnTransformer(transformers)

    n_components_type = trial.suggest_categorical('n_components_type', ['float', 'int'])

    if n_components_type == 'float':
        # Fraction of variance to preserve
        n_components = trial.suggest_float('n_components_float', 0.5, 0.999)
    else:
        # Number of components (must be ≤ min(n_samples, n_features))
        max_components = min(X_train.shape[0], X_train.shape[1])
        n_components = trial.suggest_int('n_components_int', 2, max_components)

    whiten = trial.suggest_categorical("whiten", [True, False])

    # Logistic Regression parameters
    C = trial.suggest_float("C", 1e-3, 1e3, log=True)
    penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
    solver = "liblinear" if penalty == "l1" else trial.suggest_categorical("solver", ["lbfgs", "saga"])
    max_iter_clf = trial.suggest_int("max_iter_clf",100,10000)

    random_state_clf = randint(0,1000)
    random_state_pca = randint(0,1000)
    random_state_cv = randint(0,1000)
    trial.set_user_attr("random_state_clf", random_state_clf)
    trial.set_user_attr("random_state_pca", random_state_pca)
    trial.set_user_attr("random_state_cv", random_state_cv)
    
    pipe = Pipeline([
        # ("scaler", StandardScaler()),
        ("scaler", col_transformer),
        ("pca", PCA(n_components=n_components, whiten=whiten, random_state=random_state_pca)),
        ("clf", LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=max_iter_clf, random_state=random_state_clf)),
    ])
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state_cv)
    
    recall_scores = []
    recall_scores_dropout = []

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
    df = pd.read_csv(r'project\predict_students_dropout_and_academic_success.csv')

    X = df.iloc[:, :-1]   # all rows, all columns except the last
    y = df.iloc[:,  -1]   # all rows, just the last column

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    # Run Optuna
    study = optuna.create_study(direction="maximize", study_name='no-name-4dd407ec-952a-4e52-8ad6-0eb09509a754',storage="sqlite:///example_study.db",load_if_exists=True)
    study.optimize(objective, n_trials=7)

    # Results
    print("Best trial:")
    print(f"Recall: {study.best_value}")
    print("Params:")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")

    scaler_options = {
    "standard": StandardScaler(),
    "minmax": MinMaxScaler(),
    "robust": RobustScaler(),
    "ordinal": OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
    "none": "passthrough",  # Use 'passthrough' instead of None
    }

    # Rebuild transformers from study.best_params
    transformers = []
    for key, value in study.best_params.items():
        if key.startswith("scaler_"):
            column_name = key.replace("scaler_", "")
            scaler = scaler_options[value]
            transformers.append((f"{column_name}_scaler", scaler, [column_name]))
    # Create the ColumnTransformer
    col_transformer = ColumnTransformer(transformers)


    n_components = study.best_params["n_components_int"] if study.best_params['n_components_type'] == "int" else study.best_params["n_components_float"]
    whiten = study.best_params["whiten"]
    # random_state_pca = study.best_params["random_state_pca"]
    C = study.best_params["C"]
    penalty = study.best_params["penalty"]
    solver = study.best_params["solver"]
    max_iter_clf = study.best_params["max_iter_clf"]
    # random_state_clf = study.best_params["random_state_clf"]

    pipe = Pipeline([
        # ("scaler", StandardScaler()),
        ("scaler", col_transformer),
        ("pca", PCA(n_components=n_components, whiten=whiten)),
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
