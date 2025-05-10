import pandas as pd
import scipy
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("D:/project/fake_job/static/fake_job_postings.csv")

# Select relevant features
df = df[['title', 'location', 'description', 'requirements',
         'telecommuting', 'has_company_logo', 'has_questions', 'fraudulent']]

# Fill missing values in text columns
text_cols = ['title', 'location', 'description', 'requirements']
for col in text_cols:
    df[col] = df[col].fillna('')

# Combine text features
df['combined_text'] = df['title'] + ' ' + df['location'] + ' ' + df['description'] + ' ' + df['requirements']

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_text = tfidf.fit_transform(df['combined_text'])

# Binary features
X_numeric = df[['telecommuting', 'has_company_logo', 'has_questions']].values

# Combine text and numeric features
X = scipy.sparse.hstack([X_text, X_numeric])
y = df['fraudulent']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and Evaluate
y_pred = model.predict(X_test)
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "Model.pkl")
joblib.dump(tfidf, "Vectorizer.pkl")

print("Model and vectorizer saved successfully.")
