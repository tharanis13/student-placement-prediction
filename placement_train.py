# placement_train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load dataset
df = pd.read_csv(r"C:\mlproject\placementdata.csv")

# Clean column names: strip spaces
df.columns = df.columns.str.strip()

# Categorical columns
categorical_cols = ['Internships', 'Projects', 'Workshops/Certifications',
                    'SoftSkillsRating', 'ExtracurricularActivities', 'PlacementTraining']

# Encode categorical variables
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Encode target
target_le = LabelEncoder()
df['PlacementStatus'] = target_le.fit_transform(df['PlacementStatus'])

# Features and target
X = df.drop(['StudentID', 'PlacementStatus'], axis=1)
y = df['PlacementStatus']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model, scaler, and encoders
joblib.dump(model, r"C:\mlproject\placement_model.pkl")
joblib.dump(scaler, r"C:\mlproject\scaler.pkl")
joblib.dump(le_dict, r"C:\mlproject\le_dict.pkl")
joblib.dump(target_le, r"C:\mlproject\target_le.pkl")

print("Model, scaler, and encoders saved successfully!")
