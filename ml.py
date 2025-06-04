import pandas as pd

# Load raw data
df_raw = pd.read_csv("heart_disease_uci.csv")
print(df_raw.head())

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Drop irrelevant columns
df = df_raw.drop(columns=["id", "dataset"])

# Encode categorical columns
categorical_cols = df.select_dtypes(include="object").columns
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Save processed data
df.to_csv("processed_data.csv", index=False)
df.head()


import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Target and features
X = df.drop(columns=["num"])
y = (df["num"] > 0).astype(int)  # Convert to binary classification

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print(report)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
