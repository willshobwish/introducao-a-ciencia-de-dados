import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

if __name__ == "__main__":
    mlflow.set_tracking_uri("file:./mlruns")

    df = pd.read_csv(r'predict_students_dropout_and_academic_success.csv')

    for version in ["Default", "Drop enrolled"]:
        # Filter dataset if needed
        if version == "Drop enrolled":
            df = df[df["Target"] != "Enrolled"]

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        le = LabelEncoder()
        y = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

        model = LogisticRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        f1_score_per_class = f1_score(y_test, preds, average=None)
        dropout_index = le.transform(["Dropout"])[0]
        dropout_f1_score = f1_score_per_class[dropout_index]

        print(version)
        print("Dropout F1 Score:", dropout_f1_score)

        # MLflow logging
        with mlflow.start_run(run_name=f"Logistic regression {version}"):
            mlflow.log_param("model", "LogisticRegression")
            mlflow.log_param("version", version)
            mlflow.log_metric("dropout_f1_score", dropout_f1_score)
            mlflow.sklearn.log_model(model, "model")
