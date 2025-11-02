# 💬 AI Sentiment Analysis Classifier

A lightweight **sentiment analysis model** built in Python using **Naive Bayes**.  
It classifies text into **positive** or **negative** sentiment using a small custom dataset — ideal for learning the basics of **Natural Language Processing (NLP)**.

---

## 🚀 Overview
The **Sentiment Analysis Classifier** uses a simple **text classification pipeline** to analyze whether a sentence expresses a positive or negative emotion.

It demonstrates:
- Text preprocessing (Bag of Words + TF-IDF)
- Model training and evaluation
- Saving and reusing trained models with `joblib`

---

## 🧠 How It Works
1. Builds a **tiny sample dataset** of positive and negative sentences.  
2. Splits the dataset into **training** and **testing** sets.  
3. Trains a **Naive Bayes** model using a Scikit-learn **Pipeline**:
   - `CountVectorizer()` → converts text to numerical features  
   - `TfidfTransformer()` → applies term weighting  
   - `MultinomialNB()` → classifies sentiment  
4. Evaluates performance with accuracy and classification reports.  
5. Saves the trained model as `sentiment_pipeline.joblib`.

---

## 🧩 Features
✅ Tiny built-in dataset for quick testing  
✅ End-to-end ML pipeline (Vectorization → Training → Evaluation)  
✅ Model persistence with `joblib`  
✅ Easy to extend with your own text data  
✅ Ideal for NLP beginners  

---

## 🖥️ Example Output
```bash
Accuracy: 1.0
              precision    recall  f1-score   support

         neg       1.00      1.00      1.00         1
         pos       1.00      1.00      1.00         1

    accuracy                           1.00         2
   macro avg       1.00      1.00      1.00         2
weighted avg       1.00      1.00      1.00         2

Saved pipeline to sentiment_pipeline.joblib
Example prediction: ['pos']
⚙️ Installation
bash
Copy code
# Clone the repository
git clone https://github.com/your-username/sentiment_analysis.git

# Navigate to the folder
cd sentiment_analysis

# Install dependencies
pip install -r requirements.txt
▶️ Usage
bash
Copy code
python sentiment_analysis.py
This will:

Train and evaluate the sentiment classifier

Save the trained pipeline

Print an example prediction for a test sentence

🛠️ Tech Stack
Language: Python 🐍

Libraries:

scikit-learn

joblib

🧑‍💻 Author
Muhammad Mujab Bin Muzahar
AI & Robotics Enthusiast | Web Developer | CEO @ MMBM
📫 mmujab12@gmail.com
