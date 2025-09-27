# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Import Required Libraries

Import pandas, matplotlib, and sklearn modules for data handling, visualization, and ML.

Step 2: Load Dataset

Read Employee.csv using pandas.read_csv().

Step 3: Identify Target and Features

Target column → left (1 = employee left, 0 = stayed).

Features → satisfaction_level, last_evaluation, number_project, average_montly_hours, time_spend_company, Work_accident, promotion_last_5years, Departments, salary.

Step 4: Preprocess Data

Encode categorical variables (Departments, salary) into numeric values using Label Encoding.

Ensure no missing values exist.

Step 5: Split Dataset

Divide dataset into training set (70%) and testing set (30%) using train_test_split.

Step 6: Build Decision Tree Classifier

Initialize DecisionTreeClassifier with criterion = "entropy" (or "gini") and a maximum depth to prevent overfitting.

Train (fit) the model using training data.

Step 7: Make Predictions

Use the trained model to predict on the test data.

Step 8: Evaluate the Model

Measure performance using:

Accuracy Score

Confusion Matrix

Classification Report (Precision, Recall, F1-Score).

Step 9: Visualize the Decision Tree

Plot the decision tree using sklearn.tree.plot_tree to understand the decision rules.

## Program:
```python
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: M syed rasshid husain
RegisterNumber: 25009038

# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load dataset
df = pd.read_csv(r"C:\Users\israv\Downloads\employee.csv")

# Step 3: Preprocessing
# Target column is 'left'
X = df.drop("left", axis=1)
y = df["left"]

# Encode categorical variables (Departments, salary)
for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

# Step 4: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 5: Train Decision Tree Classifier
clf = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Step 6: Predictions
y_pred = clf.predict(X_test)

# Step 7: Evaluation
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 8: Visualization
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=X.columns, class_names=["Stay","Left"], filled=True)
plt.show()

```

## Output:
<img width="514" height="341" alt="Screenshot 2025-09-26 202417" src="https://github.com/user-attachments/assets/b88837e8-5c04-4c31-8b3c-37ce5cd0c150" />
<img width="1395" height="664" alt="Screenshot 2025-09-26 202449" src="https://github.com/user-attachments/assets/7c523f60-ef42-4f27-829d-faf38fe71555" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
