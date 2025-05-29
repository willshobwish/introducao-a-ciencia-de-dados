from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier  # or any estimator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from random import randint
from sklearn.linear_model import LogisticRegression
import optuna
from optuna.trial import Trial
from sklearn.pipeline import Pipeline
from sklearn.metrics import recall_score
def objective(trial:Trial):
    
    # Example: X is your feature DataFrame, y is the label vector
    train_test_split_random_state = randint(0,1000)
    trial.set_user_attr("train_test_split_random_state", train_test_split_random_state)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=train_test_split_random_state)

    # Choose a model
    model_random_state = randint(0,1000)
    trial.set_user_attr("model_random_state", model_random_state)
    
    random_state_clf= randint(0,1000)
    trial.set_user_attr("random_state_clf", random_state_clf)

    random_forest_classifier_params = {
        'n_estimators': trial.suggest_int('random_forest_classifier_n_estimators', 10, 2000),
        'max_depth': trial.suggest_int('random_forest_classifier_max_depth', 2, 100, log=True),
        'min_samples_split': trial.suggest_int('random_forest_classifier_min_samples_split', 2, 100),
        'min_samples_leaf': trial.suggest_int('random_forest_classifier_min_samples_leaf', 1, 100),
        'max_features': trial.suggest_categorical('random_forest_classifier_max_features', ['sqrt', 'log2', None]),
    }

    whiten = trial.suggest_categorical("whiten", [True, False])

    # Logistic Regression parameters
    C = trial.suggest_float("C", 1e-3, 1e3, log=True)
    penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
    solver = "liblinear" if penalty == "l1" else trial.suggest_categorical("solver", ["lbfgs", "saga"])
    max_iter_clf = trial.suggest_int("max_iter_clf",100,10000)

    model_name = trial.suggest_categorical("model_name",['RandomForestClassifier','LogisticRegression'])
    if model_name == 'RandomForestClassifier':
        model = RandomForestClassifier(**random_forest_classifier_params, random_state=model_random_state)
    elif model_name == 'LogisticRegression':
        model = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=max_iter_clf, random_state=random_state_clf)
    # Create the RFE object and rank features
    rfe = RFE(estimator=model, n_features_to_select=trial.suggest_int("n_features_to_select",2,X_train.shape[1]))  # choose number of features to keep
    # rfe.fit(X_train, y_train)

    # Selected features mask
    pipeline = Pipeline(steps=[
        # ("preprocessor", preprocessor),
        ("feature_selection", rfe),
        ("classifier", model),
    ])
    
    pipeline.fit(X_train,y_train)
    
    preds = pipeline.predict(X_test)
    
    recall = recall_score(y_test, preds,average="weighted")
    
    selected_features = rfe.support_
    selected_columns = X.columns[selected_features]

    trial.set_user_attr("selected_features", selected_features.tolist())
    trial.set_user_attr("selected_columns", selected_columns.tolist())
    return recall

if __name__ == "__main__":
    df = pd.read_csv(r'project\predict_students_dropout_and_academic_success.csv')

    X = df.iloc[:, :-1]   # all rows, all columns except the last
    y = df.iloc[:,  -1]   # all rows, just the last column

    le = LabelEncoder()
    y = le.fit_transform(y)

    study = optuna.create_study(direction="maximize",study_name="rfe",storage="sqlite:///rfe.db",load_if_exists=True)
    study.optimize(objective, n_trials=1000)
