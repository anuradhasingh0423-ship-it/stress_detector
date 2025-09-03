import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import numpy as np

# Load original dataset with only 3 features
df = pd.read_csv('stress_data.csv').dropna()

# Simulate additional features with random data or your own collected data
np.random.seed(42)
df['Heart Rate'] = np.random.randint(40, 180, size=len(df))
df['Respiration Rate'] = np.random.randint(10, 40, size=len(df))
df['Sleep Hours'] = np.random.uniform(4, 10, size=len(df))
df['Systolic'] = np.random.randint(90, 140, size=len(df))
df['Diastolic'] = np.random.randint(60, 90, size=len(df))
df['Mood'] = np.random.randint(1, 10, size=len(df))
df['Caffeine Intake'] = np.random.randint(0, 500, size=len(df))

# Define features and target (make sure target column name is correct)
X = df.drop('Stress Level', axis=1)
y = df['Stress Level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

joblib.dump(model, 'stress_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
