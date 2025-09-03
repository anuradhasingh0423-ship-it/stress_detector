import pandas as pd

# Load dataset (change filename if needed)
df = pd.read_csv('stress_data.csv')

# Show first 5 rows
print(df.head())

# Show summary info
print(df.info())

# Show basic statistics
print(df.describe())



