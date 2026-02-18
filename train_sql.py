import pandas as pd
import sqlalchemy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Database connection string
# Adjust this string based on your database type and credentials
# Examples:
# PostgreSQL: 'postgresql://user:password@host:port/dbname'
# MySQL: 'mysql+pymysql://user:password@host:port/dbname'

db_connection_string = 'your_database_connection_string_here'

# Create database engine
engine = sqlalchemy.create_engine(db_connection_string)

# SQL query to retrieve data (adjust table and column names accordingly)
query = """
SELECT Pregnancies, Glucose, BloodPressure, BMI, Age, Outcome
FROM your_table_name
WHERE your_conditions_if_any
"""

# Load data into DataFrame
df = pd.read_sql(query, engine)

print("✅ Data loaded. Columns:", df.columns.tolist())

# Prepare features and target
X = df[["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age"]]
y = df["Outcome"]

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model to a file for deployment
model_filename = "diabetes_model.pkl"
joblib.dump(model, model_filename)

print(f"✅ Model trained and saved as {model_filename}")
