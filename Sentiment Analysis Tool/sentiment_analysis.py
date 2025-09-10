# sentiment_analysis.py
# Train a simple sentiment classifier (Naive Bayes) on a tiny sample dataset.
# Requirements: scikit-learn, joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

def build_sample():
    texts = [
        'I love this product', 'This is amazing', 'Worst purchase ever', 'Very bad experience',
        'Totally satisfied', 'Not good', 'I am happy with the quality', 'I hate it', 'Exceptional service', 'Terrible'
    ]
    labels = ['pos','pos','neg','neg','pos','neg','pos','neg','pos','neg']
    return texts, labels

def main():
    texts, labels = build_sample()
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB())
    ])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
    joblib.dump(pipeline, 'sentiment_pipeline.joblib')
    print('Saved pipeline to sentiment_pipeline.joblib')
    print('Example prediction:', pipeline.predict(['The product exceeded my expectations!']))

if __name__ == '__main__':
    main()
