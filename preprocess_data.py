import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load the dataset
df = pd.read_csv('IMDB_Dataset.csv')  # <-- Replace with your actual file name

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', str(text))
    # Remove non-letters
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Lowercase
    text = text.lower()
    # Basic tokenization (fallback)
    tokens = text.split()
    # Remove stopwords and lemmatize
    cleaned = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(cleaned)

# Apply cleaning
df['cleaned_review'] = df['review'].apply(clean_text)

# Save the cleaned dataset
df.to_csv('preprocessed_reviews.csv', index=False)

print("Preprocessing complete ✅")
print(df[['cleaned_review', 'sentiment']].head())
