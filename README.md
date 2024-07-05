# Car Price Prediction Project

## Project Overview

This project focuses on predicting car prices using machine learning algorithms. The dataset used in this project contains various features related to car specifications. The project involves data preprocessing, visualization, model building, and evaluation.

## Libraries and Tools

- **Pandas**: For data manipulation
- **NumPy**: For numerical computations
- **Matplotlib**: For data visualization
- **Seaborn**: For statistical data visualization
- **Scikit-learn**: For data preprocessing and model building
- **XGBoost**: For advanced gradient boosting
- **TensorFlow**: For potential deep learning applications

## Project Steps

1. **Reading and Exploring Data**:
    - Loaded the dataset and displayed the first few records.
    - Checked for null values and data types.

2. **Data Cleaning**:
    - Removed irrelevant columns.
    - Encoded categorical features using LabelEncoder.

3. **Data Visualization**:
    - Plotted heatmap to visualize correlations between features.
    - Displayed histograms for the distribution of features.

4. **Data Preprocessing**:
    - Label encoded the target variable.
    - Split the dataset into training and testing sets.

5. **Model Building and Evaluation**:
    - **Random Forest Regressor**:
        - Trained and evaluated the model with specified hyperparameters.
    - **Linear Regression**:
        - Built and evaluated the model.
    - **AdaBoost Regressor**:
        - Tuned hyperparameters and trained the model.
    - **XGBoost Regressor**:
        - Built and evaluated the model with specific hyperparameters.

## Results

- **Random Forest Regressor**:
    - Training R² Score: 0.9829

- **Linear Regression**:
    - Training R² Score: 0.87612

- **AdaBoost Regressor**:
    - Training R² Score: 0.99368
    - Testing R² Score: 0.96576

- **XGBoost Regressor**:
    - Training R² Score: 0.999
    - Testing R² Score: 0.961
    
## Conclusion

The project demonstrates the effective use of machine learning algorithms for car price prediction. The models were able to achieve high accuracy, showcasing the potential of machine learning in predicting car prices.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Osama-Abo-Bakr/car-price-prediction.git
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook or script to see the results.

## Contact

Feel free to reach out if you have any questions or suggestions.

- **Email**: [Gmail](mailto:osamaoabobakr12@gmail.com)
- **LinkedIn**: [LinkedIn](https://linkedin.com/in/osama-abo-bakr-293614259/)

---

### Sample Code (for reference)

```python
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb

# Reading the data
data = pd.read_csv('path_to_your_data.csv')
data.drop(columns="Car_Name", inplace=True)

# Encoding categorical variables
label_encoder = LabelEncoder()
columns = ["Fuel_Type", "Seller_Type", "Transmission"]
for col in columns:
    data[col] = label_encoder.fit_transform(data[col])

# Splitting data
X = data.drop(columns="Selling_Price")
Y = data["Selling_Price"]
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, random_state=42)

# Random Forest Regressor model
RF_reg = RandomForestRegressor(n_estimators=50, max_depth=100)
RF_reg.fit(x_train, y_train)
print("Random Forest - Training R² Score:", r2_score(y_train, RF_reg.predict(x_train)))

# Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)
print("Linear Regression - Training R² Score:", r2_score(y_train, lin_reg.predict(x_train)))

# AdaBoost Regressor model
Adaboost_reg = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=100, min_samples_split=5, min_samples_leaf=6, random_state=42), n_estimators=100, learning_rate=0.2)
Adaboost_reg.fit(x_train, y_train)
print("AdaBoost - Training R² Score:", Adaboost_reg.score(x_train, y_train))
print("AdaBoost - Testing R² Score:", Adaboost_reg.score(x_test, y_test))

# XGBoost Regressor model
model_xgb = xgb.XGBRegressor(n_estimators=100, max_depth=20, learning_rate=0.2, min_child_weight=0.1, random_state=42)
model_xgb.fit(x_train, y_train)
print("XGBoost - Training R² Score:", model_xgb.score(x_train, y_train))
print("XGBoost - Testing R² Score:", model_xgb.score(x_test, y_test))
```

Feel free to edit and customize the README file further to match your preferences and project details.
