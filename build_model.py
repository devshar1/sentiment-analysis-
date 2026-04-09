import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. Load the preprocessed data
df = pd.read_csv('preprocessed_reviews.csv')

# 2. Define features and labels
X = df['cleaned_review']
y = df['sentiment']

# 3. Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer(max_features=3000)
X_vectorized = vectorizer.fit_transform(X)

# 4. Split into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# 5. Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 6. Predict on test data
y_pred = model.predict(X_test)

# 7. Evaluate results
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy Score:", accuracy_score(y_test, y_pred))
