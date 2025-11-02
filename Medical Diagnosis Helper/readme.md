# 🏥 AI Medical Diagnosis Classifier

A **machine learning–powered toy medical diagnosis system** that classifies patients into potential health categories using synthetic data and a Random Forest model.

---

## 🚀 Overview
The **Medical Diagnosis Classifier** uses **synthetic data generation** and **AI-based modeling** to simulate a diagnostic system that predicts whether a patient may have:
- **Disease A**
- **Disease B**
- or is **Healthy**

It’s designed for **educational and experimental purposes**, showcasing how ML can be applied to healthcare data.

---

## 🧠 How It Works
1. Generates a **synthetic dataset** using `make_classification()` from scikit-learn.  
2. Trains a **Random Forest Classifier** on the data.  
3. Evaluates model accuracy and prints a **classification report**.  
4. Saves the trained model to a file (`medical_diagnosis_model.joblib`) for reuse.  
5. Displays **sample predictions** for quick testing.

---

## 📊 Example Output
```bash
Accuracy: 0.91
              precision    recall  f1-score   support

    Disease_A       0.92      0.90      0.91       52
    Disease_B       0.90      0.91      0.91       56
       Healthy       0.91      0.92      0.91       52

    accuracy                           0.91      160
   macro avg       0.91      0.91      0.91      160
weighted avg       0.91      0.91      0.91      160

Saved model to medical_diagnosis_model.joblib
Sample predictions: ['Healthy' 'Disease_B' 'Disease_A']
🧩 Features
✅ Synthetic dataset generation for experimentation
✅ Multi-class classification (3 labels)
✅ Random Forest–based prediction model
✅ Built-in model saving with joblib
✅ Simple and lightweight — perfect for learning ML fundamentals

🛠️ Tech Stack
Language: Python 🐍

Libraries:

numpy

pandas

scikit-learn

joblib

⚙️ Installation
bash
Copy code
# Clone the repository
git clone https://github.com/your-username/medical_diagnosis.git

# Navigate to project folder
cd medical_diagnosis

# Install required libraries
pip install -r requirements.txt
▶️ Usage
bash
Copy code
python medical_diagnosis.py
This will:

Generate a dataset

Train and test the model

Print performance metrics

Save the trained model to medical_diagnosis_model.joblib

🧑‍💻 Author
Muhammad Mujab Bin Muzahar
AI & Robotics Enthusiast | Web Developer | CEO @ MMBM
📫 mmujab12@gmail.com
