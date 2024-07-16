import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve

# Load the datasets
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('Copy of test_data.csv')

# Display the first few rows of the train dataset
print("Train Data Head:")
print(train_data.head())

# Display basic info and column names for train dataset
print("\nTrain Data Info:")
print(train_data.info())
print(train_data.columns)

# Handling missing values in the train dataset
numeric_columns = train_data.select_dtypes(include=['float64', 'int64']).columns
train_data[numeric_columns] = train_data[numeric_columns].fillna(train_data[numeric_columns].mean())

# Encoding categorical variables in the train dataset
label_encoders = {}
for column in train_data.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    train_data[column] = label_encoders[column].fit_transform(train_data[column])

# Scaling numerical features in the train dataset
scaler = MinMaxScaler()
train_data[numeric_columns] = scaler.fit_transform(train_data[numeric_columns])

# Display the first few rows of the test dataset
print("Test Data Head:")
print(test_data.head())

# Display basic info and column names for test dataset
print("\nTest Data Info:")
print(test_data.info())
print(test_data.columns)

# Handling missing values in the test dataset
numeric_columns_test = test_data.select_dtypes(include=['float64', 'int64']).columns
test_data[numeric_columns_test] = test_data[numeric_columns_test].fillna(test_data[numeric_columns_test].mean())

for column in test_data.select_dtypes(include=['object']).columns:
    if column in label_encoders:
        test_data[column] = label_encoders[column].transform(test_data[column])

# Scaling numerical features in the test dataset
test_data[numeric_columns_test] = scaler.transform(test_data[numeric_columns_test])

# Exploratory Data Analysis (EDA) on the train dataset

## Basic Statistics
print("\nTrain Data Description:")
print(train_data.describe())

## Univariate Analysis
plt.figure(figsize=(10, 6))
sns.histplot(train_data['Age'], bins=20, kde=True)
plt.title('Age Distribution (Train Data)')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

## Bivariate Analysis
plt.figure(figsize=(10, 6))
sns.boxplot(x='Has a car', y='Income', data=train_data)
plt.title('Income vs Car Ownership (Train Data)')
plt.show()

## Correlation Analysis
plt.figure(figsize=(12, 10))
sns.heatmap(train_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix (Train Data)')
plt.show()

# Feature Selection with SelectKBest on the train dataset
X_train_full = train_data.drop('Is high risk', axis=1)
y_train_full = train_data['Is high risk']

# Using chi-squared statistical test to select top 10 features
selector = SelectKBest(score_func=chi2, k=10)
X_train_full_new = selector.fit_transform(X_train_full, y_train_full)

# Getting selected feature names
selected_features = X_train_full.columns[selector.get_support(indices=True)]

print("\nSelected Features:")
print(selected_features)

# Preprocessing the test dataset using the same selected features
X_test_full = test_data.drop('Is high risk', axis=1)
y_test_full = test_data['Is high risk']
X_test_full = X_test_full[selected_features]

# Splitting the train dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_full[selected_features], y_train_full, test_size=0.2, random_state=42)

# Model Development
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

# Training and Evaluating Models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    
    print(f"{name} Performance on Validation Data:")
    print(f"Accuracy: {accuracy_score(y_val, y_val_pred)}")
    print(f"Precision: {precision_score(y_val, y_val_pred)}")
    print(f"Recall: {recall_score(y_val, y_val_pred)}")
    print(f"F1 Score: {f1_score(y_val, y_val_pred)}")
    print(f"Confusion Matrix:\n {confusion_matrix(y_val, y_val_pred)}")
    print(f"ROC AUC: {roc_auc_score(y_val, y_val_pred)}\n")

    # Evaluating on Test Data
    y_test_pred = model.predict(X_test_full)
    print(f"{name} Performance on Test Data:")
    print(f"Accuracy: {accuracy_score(y_test_full, y_test_pred)}")
    print(f"Precision: {precision_score(y_test_full, y_test_pred)}")
    print(f"Recall: {recall_score(y_test_full, y_test_pred)}")
    print(f"F1 Score: {f1_score(y_test_full, y_test_pred)}")
    print(f"Confusion Matrix:\n {confusion_matrix(y_test_full, y_test_pred)}")
    print(f"ROC AUC: {roc_auc_score(y_test_full, y_test_pred)}\n")

    # Predicting Credit Card Approval using the best model
    if name == 'Random Forest':  # Example: using Random Forest as the best model
        y_test_best = model.predict(X_test_full)

        # Feature Importance
        importances = model.feature_importances_
        features = selected_features
        indices = np.argsort(importances)[::-1]

        print("Feature importances:")
        for f in range(X_train_full[selected_features].shape[1]):
            print(f"{features[indices[f]]}: {importances[indices[f]]}")

        # Plotting feature importances
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(X_train_full[selected_features].shape[1]), importances[indices], align="center")
        plt.xticks(range(X_train_full[selected_features].shape[1]), features[indices], rotation=90)
        plt.xlim([-1, X_train_full[selected_features].shape[1]])
        plt.show()

        # Recommendations based on findings
        print("Recommendations:")
        print("1. Focus on applicants with higher incomes and longer employment lengths.")
        print("2. Implement stricter criteria for applicants with lower income and shorter employment durations.")

# Predictions on Test Data using Random Forest
prediction_df = pd.DataFrame({
    'Actual': y_test_full,
    'Predicted': y_test_best
})
print(f"Predictions using Random Forest:")
print(prediction_df)
