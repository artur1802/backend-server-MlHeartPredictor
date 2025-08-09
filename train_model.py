# train_model.py
import json
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score

# --- Load data ---
df = pd.read_csv("heart.csv")

# --- Cholesterol cleaning ---
Cholesterol_mean = df.loc[df["Cholesterol"] != 0, "Cholesterol"].mean()
df["Cholesterol"] = df["Cholesterol"].replace(0, Cholesterol_mean)

# --- Encode/one-hot as in your pipeline ---
df["Sex"] = df["Sex"].map({"M": 1, "F": 0})
le = LabelEncoder()
df["ExerciseAngina"] = le.fit_transform(df["ExerciseAngina"])

df = pd.get_dummies(df, columns=["ChestPainType"], prefix="chestPain")
df = pd.get_dummies(df, columns=["ST_Slope"], prefix="ST_Slope")
df = pd.get_dummies(df, columns=["RestingECG"], prefix="RestingECG")

# --- Features & target ---
features_to_scale = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
X = df[
    [
        'Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak',
        'Sex', 'FastingBS', 'ExerciseAngina',
        'chestPain_ASY', 'chestPain_ATA', 'chestPain_NAP', 'chestPain_TA',
        'ST_Slope_Down', 'ST_Slope_Flat', 'ST_Slope_Up',
        'RestingECG_LVH', 'RestingECG_Normal', 'RestingECG_ST'
    ]
]
y = df['HeartDisease']

# --- Fit scaler on training data (or full data if you prefer) ---
scaler = StandardScaler()
X[features_to_scale] = scaler.fit_transform(X[features_to_scale])

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Train RandomForest ---
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_recall = recall_score(y_test, y_pred_rf)
print(f"Random Forest - Accuracy: {rf_accuracy:.3f}")
print(f"Random Forest - Recall: {rf_recall:.3f}")

# --- Save artifacts ---
joblib.dump(rf, "model_rf.joblib")
joblib.dump(scaler, "scaler.joblib")

# Save feature order so API knows which columns to expect
feature_cols = list(X.columns)
with open("feature_cols.json", "w") as f:
    json.dump(feature_cols, f)

print("Saved model_rf.joblib, scaler.joblib, feature_cols.json")