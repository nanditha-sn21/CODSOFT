# CODSOFT
"Titanic Survival Prediction using Logistic Regression &amp; Random Forest in Python"
Got it — you just want a **README.md** file describing how you did the **Titanic Survival Prediction** project in **Google Colab**, with dataset details included.

# Titanic Survival Prediction 🚢

This project predicts the survival of passengers on the Titanic using machine learning techniques in **Google Colab**.  
It is one of the CODSOFT Internship Tasks.

---

## 📌 Project Overview
The Titanic dataset contains passenger details such as age, gender, class, and other features.  
Our goal is to build a model that predicts whether a passenger survived or not.

---

## 🛠 Tools & Libraries Used
- **Google Colab** (Python environment)
- **Pandas** – Data manipulation
- **NumPy** – Numerical calculations
- **Matplotlib / Seaborn** – Data visualization
- **Scikit-learn** – Machine learning models and evaluation

---

## 📂 Dataset
- **Source:** [Kaggle Titanic Dataset](https://www.kaggle.com/competitions/titanic/data)  
- **Files used:**
  - `train.csv` – Training dataset
  - `test.csv` – Test dataset

---

## 🚀 Steps Followed in Google Colab

### 1️⃣ Setup Environment
- Open **Google Colab** and create a new notebook.
- Import required libraries:
  ```python
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  import seaborn as sns
  from sklearn.model_selection import train_test_split
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.metrics import accuracy_score, classification_report
````

### 2️⃣ Load Dataset

```python
from google.colab import files
uploaded = files.upload()
df = pd.read_csv('train.csv')
df.head()
```

### 3️⃣ Data Preprocessing

* Handle missing values (Age, Embarked).
* Convert categorical features into numerical (Sex, Embarked).
* Drop unnecessary columns (`Name`, `Ticket`, `Cabin`).

### 4️⃣ Train-Test Split

```python
X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 5️⃣ Model Training

```python
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
```

### 6️⃣ Evaluation

```python
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

---

## 📊 Results

* **Accuracy:** \~0.80 (80%)
* Feature importance shows **Sex**, **Pclass**, and **Fare** as the top predictors.

---

## 📌 How to Run This Project

1. Download the Titanic dataset from [Kaggle](https://www.kaggle.com/competitions/titanic/data).
2. Open Google Colab and upload the CSV files.
3. Run the notebook cells step-by-step.

---

## 📜 License

This project is created for educational purposes as part of **CODSOFT Internship**.

---

```

---

If you want, I can also **add a section with Colab and Kaggle dataset links** so anyone can open and run it instantly from your README.  
Do you want me to add that?
```
