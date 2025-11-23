# Experiment 5: COVID-19 Daily Cases Forecasting (Bivariate Analysis)
**AIM:**

To perform bivariate analysis on COVID-19 data and study the relationship between daily confirmed cases and daily deaths, using Linear Regression for prediction and simple real-time forecasting.

**ALGORITHM / PROCEDURE:**

1)Import the required Python libraries: pandas, numpy, matplotlib, and sklearn modules for model building and evaluation.

2)Load the COVID-19 dataset (CSV file) into a pandas DataFrame.

3)Select a specific country’s data (e.g., India or USA) for analysis.

4)Perform bivariate analysis by calculating the correlation between daily cases and daily deaths.

5)Visualize the relationship between daily cases and daily deaths using a scatter plot.

6)Define the input variable (X = Daily Cases) and the output variable (y = Daily Deaths).

7)Split the dataset into training and testing parts using the train_test_split() function.

8)Create and train a Linear Regression model using the training data.

9)Predict daily deaths for the test dataset using the trained model.

10)Evaluate the model performance by calculating R² (coefficient of determination) and RMSE (Root Mean Squared Error).

11)Plot a graph comparing actual deaths versus predicted deaths to visualize model accuracy.

12)Plot daily cases and daily deaths over time to observe the time trend and lag between them.

13)Generate a range of possible future case numbers and predict corresponding deaths using the trained model.



**Program**
## NAME: Vignesh Raaj
## REG NUMBER: 212223230239

```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os

# Load Dataset

url = "https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv"

try:
    data = pd.read_csv(url)
    print(" Dataset loaded successfully from online source.")
except Exception as e:
    print(" Online dataset could not be loaded. Using local fallback sample dataset.")
    # Fallback local sample (if offline)
    fallback_url = "https://raw.githubusercontent.com/datasets/covid-19/main/data/countries-aggregated.csv"
    data = pd.read_csv(fallback_url)

# Filter for a Country

country = "India"  # You can change to "USA", "Brazil"
country_data = data[data['Country'] == country].copy()

# Convert Date to datetime
country_data['Date'] = pd.to_datetime(country_data['Date'])

# Compute daily new cases and deaths
country_data['Daily_Cases'] = country_data['Confirmed'].diff().fillna(0)
country_data['Daily_Deaths'] = country_data['Deaths'].diff().fillna(0)

# Display first few rows
print("\nSample of processed data:")
print(country_data.head())

# Bivariate Visualization

plt.figure(figsize=(7,5))
plt.scatter(country_data['Daily_Cases'], country_data['Daily_Deaths'], alpha=0.5, color='purple')
plt.title(f'Bivariate Analysis: {country} COVID-19 Daily Cases vs Deaths')
plt.xlabel('Daily Cases')
plt.ylabel('Daily Deaths')
plt.grid(True)
plt.show()

# Correlation

corr = country_data['Daily_Cases'].corr(country_data['Daily_Deaths'])
print(f"\n Correlation between Daily Cases and Deaths in {country}: {corr:.3f}")

#Prepare Data for Regression

X = country_data[['Daily_Cases']]
y = country_data['Daily_Deaths']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


# Train Linear Regression Model

model = LinearRegression()
model.fit(X_train, y_train)

# Predict & Evaluate

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n Model Evaluation for {country}:")
print(f"   R² Score  = {r2:.3f}")
print(f"   RMSE      = {rmse:.3f}")

# Actual vs Predicted Plot

plt.figure(figsize=(10,5))
plt.plot(country_data['Date'][-len(y_test):], y_test, label='Actual Deaths', color='blue')
plt.plot(country_data['Date'][-len(y_pred):], y_pred, label='Predicted Deaths', color='red', linestyle='--')
plt.title(f'{country}: COVID-19 Actual vs Predicted Daily Deaths')
plt.xlabel('Date')
plt.ylabel('Daily Deaths')
plt.legend()
plt.grid(True)
plt.show()

# Forecast Future Deaths (Simple Trend)

future_cases = np.linspace(X['Daily_Cases'].max()*0.5, X['Daily_Cases'].max()*1.2, 10).reshape(-1, 1)
future_pred = model.predict(future_cases)

print("\n Forecast of Future Deaths based on Case Count:")
for c, d in zip(future_cases.flatten(), future_pred):
    print(f"   Predicted deaths for {int(c)} cases ≈ {d:.1f}")

# Plot Both Cases and Deaths Over Time

plt.figure(figsize=(10,5))
plt.plot(country_data['Date'], country_data['Daily_Cases'], label='Daily Cases', color='green')
plt.plot(country_data['Date'], country_data['Daily_Deaths'], label='Daily Deaths', color='orange')
plt.title(f'{country}: COVID-19 Daily Cases and Deaths Over Time')
plt.xlabel('Date')
plt.ylabel('Count')
plt.legend()
plt.grid(True)
plt.show()
```


**Output**

<img width="622" height="711" alt="512492458-c681ef87-9c0e-4607-8a8e-98bb1dd78397" src="https://github.com/user-attachments/assets/da225bb1-6719-4b79-82f2-6249295ae8a2" />

<img width="830" height="534" alt="512492567-845b1b1c-19a5-4713-86d7-e127e1cd9558" src="https://github.com/user-attachments/assets/4b56c97f-9770-4bc9-acf0-d39596ec3465" />

<img width="1285" height="667" alt="512492645-99b6c2aa-ccaa-4e4f-94c7-d2b33bb1d54f" src="https://github.com/user-attachments/assets/e3c061d3-9eeb-479f-b4ac-4c69097bddbb" />


**Result**
The bivariate analysis shows a strong positive correlation between daily COVID-19 cases and deaths.
The linear regression model successfully predicts deaths from case counts with good accuracy (R² ≈ 0.8).
This experiment demonstrates how linear models can be used for simple real-time forecasting in public health analytics.
Display the predicted future deaths along with correlation, R², and RMSE values.
