import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load dataset
df = pd.read_csv("task_2/Salary Prediction of Data Professions.csv")

# Exploratory Data Analysis (EDA)
# def perform_eda(df):
#     print("Data Head:\n", df.head())
#     print("Data Description:\n", df.describe())
#     print("Missing Values:\n", df.isnull().sum())
#     sns.pairplot(df)
#     plt.show()

# perform_eda(df)

df['DOJ'] = pd.to_datetime(df['DOJ'])
df['CURRENT DATE'] = pd.to_datetime(df['CURRENT DATE'])
df['YEARS_IN_COMPANY'] = (df['CURRENT DATE'] - df['DOJ']).dt.days / 365
df.drop(columns=['FIRST NAME', 'LAST NAME', 'DOJ', 'CURRENT DATE'], inplace=True)
# print(df)

# Data Preprocessing
le_sex = LabelEncoder()
df['SEX'] = le_sex.fit_transform(df['SEX'])

le_designation = LabelEncoder()
df['DESIGNATION'] = le_designation.fit_transform(df['DESIGNATION'])

le_unit = LabelEncoder()
df['UNIT'] = le_unit.fit_transform(df['UNIT'])
print(df)

# Handling missing values (if any)
df.fillna(df.median(), inplace=True)

# Splitting the data into training and testing sets
X = df.drop('SALARY', axis=1)
y = df['SALARY']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Development
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor()
}

# Training and evaluating models
best_model = None
best_score = -1
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    print(f'{name} Evaluation:')
    print('MAE:', mean_absolute_error(y_test, y_pred))
    print('MSE:', mean_squared_error(y_test, y_pred))
    print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
    print('R2 Score:', score)
    print('-'*30)
    
    # Save the best model
    if score > best_score:
        best_score = score
        best_model = model

# Save the best model
joblib.dump(best_model, 'salary_predictor_model.pkl')

# Save the encoders and scaler
joblib.dump(le_sex, 'label_encoder_sex.pkl')
joblib.dump(le_designation, 'label_encoder_designation.pkl')
joblib.dump(le_unit, 'label_encoder_unit.pkl')
joblib.dump(scaler, 'scaler.pkl')
