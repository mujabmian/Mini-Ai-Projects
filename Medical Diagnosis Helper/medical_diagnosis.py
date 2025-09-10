# medical_diagnosis.py
# Toy medical diagnosis classifier using synthetic data.
# Requirements: scikit-learn, pandas, numpy, joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

def build_dataset(n=500):
    X, y = make_classification(n_samples=n, n_features=8, n_informative=5, n_redundant=1, n_classes=3, random_state=42)
    mapping = {0: 'Disease_A', 1: 'Disease_B', 2: 'Healthy'}
    y_named = [mapping[int(v)] for v in y]
    cols = [f'feat_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df['label'] = y_named
    return df

def main():
    df = build_dataset(800)
    X = df.drop('label', axis=1)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
    joblib.dump(model, 'medical_diagnosis_model.joblib')
    print('Saved model to medical_diagnosis_model.joblib')
    sample = X_test.iloc[0:3]
    print('Sample predictions:', model.predict(sample))

if __name__ == '__main__':
    main()
