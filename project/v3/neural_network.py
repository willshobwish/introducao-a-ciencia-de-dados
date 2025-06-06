import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import optuna
from sklearn.metrics import f1_score

# Load your dataset
df = pd.read_csv(r"project\v3\predict_students_dropout_and_academic_success.csv")  # replace with your path

# Encode the target
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["Target"])
X = df.drop(columns=["Target"])

# Detect types
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Preprocess
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])

X_processed = preprocessor.fit_transform(X)
# print(X_processed)
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, stratify=y)

# Convert to PyTorch Tensors
X_train_tensor = torch.tensor(X_train.toarray() if hasattr(X_train, "toarray") else X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.toarray() if hasattr(X_test, "toarray") else X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

class Net(nn.Module):
    def __init__(self, input_dim, trial, num_classes):
        super(Net, self).__init__()
        self.layers = nn.Sequential()
        n_layers = trial.suggest_int("n_layers", 1, 10)

        in_dim = input_dim
        for i in range(n_layers):
            out_dim = trial.suggest_int(f"n_units_l{i}", 32, 1024)
            self.layers.append(nn.Linear(in_dim, out_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(trial.suggest_float(f"dropout_l{i}", 0.2, 0.5)))
            in_dim = out_dim

        self.output = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        x = self.layers(x)
        return self.output(x)

def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(np.unique(y))
    batch_size = trial.suggest_categorical("batch_size", [8,16, 32, 64, 128])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    model = Net(X_train_tensor.shape[1], trial, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Training loop
    for epoch in range(50):  # keep short for tuning
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()

    # y_true and y_pred must be numpy arrays of labels
    y_true_all = []
    y_pred_all = []

    # Evaluation
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)
            y_true_all.extend(yb.cpu().numpy())
            y_pred_all.extend(preds.cpu().numpy())

    # Compute per-class F1 score
    f1_score_per_class = f1_score(y_true_all, y_pred_all, average=None)

    # Transform 'Dropout' to its encoded index
    dropout_index = label_encoder.transform(["Dropout"])[0]
    dropout_f1_score = f1_score_per_class[dropout_index]

    # You can return dropout_f1_score instead of accuracy for tuning
    return dropout_f1_score

study = optuna.create_study(direction="maximize",
                            storage="sqlite:///nn.db",
                            load_if_exists=True,
                            study_name="nn")
study.optimize(objective, n_trials=1000)

print("Best trial:")
print(study.best_trial)

best_params = study.best_trial.params
final_model = Net(X_train_tensor.shape[1], study.best_trial, len(np.unique(y))).to("cuda")