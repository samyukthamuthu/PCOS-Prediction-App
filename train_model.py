
import pandas as pd
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import joblib

# Load dataset
zip_path = 'dataset/pcos_dataset.zip'

with zipfile.ZipFile(zip_path) as z:
    with z.open('PCOS_data.csv') as f:
        df = pd.read_csv(f)

# Remove unwanted columns
df = df.drop(columns=['Unnamed: 44'], errors='ignore')

# Fill missing values
imputer = SimpleImputer(strategy='mean')
df[df.columns] = imputer.fit_transform(df)

# Target variable
y = df['PCOS (Y/N)']

# Feature variables
X = df.drop(columns=['PCOS (Y/N)'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
rf = RandomForestClassifier(random_state=42)
lr = LogisticRegression(max_iter=1000)
xgb = XGBClassifier(eval_metric='logloss')

# Hybrid Voting Classifier
model = VotingClassifier(
    estimators=[
        ('rf', rf),
        ('lr', lr),
        ('xgb', xgb)
    ],
    voting='hard'
)

# Train model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)
print(classification_report(y_test, predictions))

# Save model
joblib.dump(model, 'models/hybrid_model.pkl')

print("Model saved successfully!")
