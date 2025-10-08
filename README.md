# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### 1. Load Data – Read Salary.csv into a pandas DataFrame.

#### 2. Select Features & Target – Use Level as input (X) and Salary as output (y).

#### 3. Split Data (optional) – Here dataset is small; we’ll train on full data.

#### 4. Train Model – Fit a DecisionTreeRegressor on X and y.

#### 5. Predict – Predict Salary for a given Level.

#### 6. Visualize (optional) – Plot predicted salary vs. level to see the tree regression steps.
## Program:
```python
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: M syed raashid husain
RegisterNumber: 25009038

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load dataset
data = pd.read_csv(r'C:\Users\acer\Downloads\Salary.csv.csv')

# Step 2: Select features and target
X = data[['Level']]  # Independent variable
y = data['Salary']   # Dependent variable

# Step 3: Train Decision Tree Regressor
model = DecisionTreeRegressor(random_state=0)
model.fit(X, y)

# Step 4: Predict Salary for a specific Level
level_to_predict = 6.5  # example
predicted_salary = model.predict([[level_to_predict]])
print(f"Predicted Salary for Level {level_to_predict}: {predicted_salary[0]}")

# Step 5: Visualize results
X_grid = np.arange(min(X.Level), max(X.Level), 0.01).reshape(-1,1)
plt.scatter(X, y, color='red', label='Actual Salary')
plt.plot(X_grid, model.predict(X_grid), color='blue', label='Predicted Salary')
plt.title('Decision Tree Regression Model')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.legend()
plt.show()

```

## Output:
<img width="791" height="615" alt="image" src="https://github.com/user-attachments/assets/0a46712d-e8b6-4588-af0b-b5c7590100c6" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
