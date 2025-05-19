import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
import optuna 
from sklearn.ensemble import RandomForestClassifier


# Evaluation Function
def evaluate_model(model, X_test, y_test, name="Model"):
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"\n=== {name} Metrics ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    return acc, prec, rec, f1

def rf_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 2000),
        'max_depth': trial.suggest_int('max_depth', 2, 100, log=True),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 100),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 100),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
    }

    model = RandomForestClassifier(**params, random_state=42)
    model.fit(X=X_train,y=y_train)
    return accuracy_score(y_test, model.predict(X_test))

if __name__ == "__main__":

    df = pd.read_csv("predict_students_dropout_and_academic_success.csv",delimiter=";")
    le = LabelEncoder()
    df["Target"] = le.fit_transform(df["Target"])

    # Assume df has features and df_encoded["Target"] is the encoded label
    X = df.iloc[:,:-1]
    y = df["Target"]

    # Optional: split to test performance later
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    n_trials = 200
    max_trials_callback = n_trials * 4
    storage = "postgresql://myuser:mypassword@localhost:5432/mydatabase"

    rf_study = optuna.create_study(direction="maximize", study_name="SklearnRandomForest",storage=storage, load_if_exists=True)
    rf_study.optimize(rf_objective, n_trials=n_trials,callbacks=[MaxTrialsCallback(max_trials_callback, states=(TrialState.COMPLETE,))])
    rf_best_model = RandomForestClassifier(**rf_study.best_params, random_state=42)
    rf_best_model.fit(X_train, y_train)
    evaluate_model(rf_best_model, X_test, y_test, "RandomForest")
