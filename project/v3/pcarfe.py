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

    pca_usage = trial.suggest_categorical(name="pca_usage", choices=[None, "pca"])
    rfe_usage = trial.suggest_categorical(name="rfe_usage", choices=[None, "rfe"])

    if pca_usage:
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

    if rfe_usage:
        if pca_usage:
            # Number of features to keep
            n_features = trial.suggest_int(name="rfe__n_features_to_select", low=1, high=n_components)
        else:
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

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    le = LabelEncoder()
    y = le.fit_transform(y)

    study = optuna.create_study(direction="maximize",
                                storage="sqlite:///random_forest_scaler.db",
                                load_if_exists=True,
                                study_name="no-name-4edeb741-7c80-4857-9a8a-47563ca4f188")
    study.optimize(objective, n_trials=1000)

    # Rebuild transformers from study.best_params
    # transformers = []
    # scaler_options = {  
    #     "standard": StandardScaler(),
    #     "minmax": MinMaxScaler(),
    #     "robust": RobustScaler(),
    #     "ordinal": OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
    #     "none":'passthrough'
    #     }

    # for key, value in study.best_params.items():
    #     if key.startswith("scaler_"):
    #         column_name = key.replace("scaler_", "")
    #         scaler = scaler_options[value]
    #         transformers.append((f"{column_name}_scaler", scaler, [column_name]))

    # col_transformer = ColumnTransformer(transformers)
    # pipe.fit(X_train, y_train)
    # preds = pipe.predict(X_test)
    # f1_score_per_class = f1_score(y_test, preds, average=None)
    # dropout_index = le.transform(["Dropout"])[0]
    # dropout_f1_score = f1_score_per_class[dropout_index]

    # print("Random forest (without enrolled and scaler)")
    # print("Dropout F1 Score:", dropout_f1_score)

    # print(classification_report(y_test, preds,target_names=le.classes_))

    # cm = confusion_matrix(y_test, preds)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    # disp.plot(cmap="Blues")
    # plt.title("Random forest without enrolled and scaler")
    # plt.show()