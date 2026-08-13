import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    RocCurveDisplay
)

import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("bank_customer_churn.csv")

# Drop unnecessary columns
df.drop(
    columns=["RowNumber", "CustomerId", "Surname"],
    inplace=True
)

# Features and Target
X = df.drop("Exited", axis=1)
y = df["Exited"]

# Categorical and Numerical Columns
categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(exclude=["object"]).columns

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            MinMaxScaler(),
            numerical_cols
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_cols
        )
    ]
)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Logistic Regression Pipeline
model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation
print(classification_report(y_test, y_pred))

print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

# ROC Curve
RocCurveDisplay.from_predictions(y_test, y_prob)
plt.show()

# Export Risk Scores
results = X_test.copy()

results["Actual_Churn"] = y_test.values
results["Predicted_Churn"] = y_pred
results["Churn_Risk_Score"] = y_prob

results.to_csv(
    "churn_predictions.csv",
    index=False
)

print("Predictions exported successfully.")
