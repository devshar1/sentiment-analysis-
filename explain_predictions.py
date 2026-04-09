import pandas as pd
import pickle
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# 1. Load preprocessed data
df = pd.read_csv('preprocessed_reviews.csv')
X = df['cleaned_review']
y = df['sentiment']

# 2. Vectorize text
vectorizer = TfidfVectorizer(max_features=3000)
X_vectorized = vectorizer.fit_transform(X)

# 3. Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train model again (optional: or load from previous step)
model = LogisticRegression(max_iter=1000)
model.fit(vectorizer.transform(X_train), y_train)

# 5. Create a pipeline for LIME
c = make_pipeline(vectorizer, model)

# 6. Set up LIME explainer
explainer = LimeTextExplainer(class_names=['negative', 'positive'])

# 7. Pick a test review to explain
i = 10  # Change this index to explain different reviews
review_text = X_test.iloc[i]
print("\nReview Text ⬇️:")
print(review_text)

# 8. Explain the prediction
exp = explainer.explain_instance(review_text, c.predict_proba, num_features=10)
print("\nPrediction Probabilities:", c.predict_proba([review_text]))
print("Model Prediction:", c.predict([review_text]))
print("\nWord Importance:")
exp.show_in_notebook(text=review_text)
# Add at the end of your script:
exp.save_to_file('lime_explanation.html')
