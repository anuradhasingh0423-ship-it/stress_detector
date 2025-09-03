import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('stress_data.csv')
print("Columns in dataset:", df.columns)

# Drop rows with missing values (or handle them accordingly)
df = df.dropna()

# Use exact column name with space and capitalization
X = df.drop('Stress Level', axis=1)
y = df['Stress Level']

# Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features for better model performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
