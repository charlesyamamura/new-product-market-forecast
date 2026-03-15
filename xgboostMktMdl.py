import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBRegressor


# -----------------------------
# 1. Load dataset
# -----------------------------

df = pd.read_excel("data1319.xlsx")


# -----------------------------
# 2. Define target
# -----------------------------

y = df["share"]


# -----------------------------
# 3. Define features
# -----------------------------

X = df.drop(columns=["share"])


# -----------------------------
# 4. Encode categorical features
# -----------------------------

for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])


# -----------------------------
# 5. Train-test split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# 6. Define XGBoost model
# -----------------------------

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)


# -----------------------------
# 7. Train model
# -----------------------------

model.fit(X_train, y_train)


# -----------------------------
# 8. Predictions
# -----------------------------

predictions = model.predict(X_test)


# -----------------------------
# 9. Evaluate model
# -----------------------------

rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("RMSE:", rmse)
print("R²:", r2)


# -----------------------------
# 10. Predict new data
# -----------------------------

# Example: predict the first 5 rows
new_predictions = model.predict(X.iloc[:5])

print("Example predictions:")
print(new_predictions)

# K-Fold cross validation
from sklearn.model_selection import cross_val_score
import numpy as np

scores = cross_val_score(model, X, y, cv=5, scoring="r2")

print("R2 scores:", scores)
print("Mean R2:", np.mean(scores))