import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the training data
train_data = pd.read_csv('train.csv')

# Feature selection
# We're assuming 'GrLivArea' for square footage, 'BedroomAbvGr' for bedrooms, and 'FullBath' + 'HalfBath' for bathrooms
train_data['TotalBath'] = train_data['FullBath'] + (train_data['HalfBath'] * 0.5)

# Select relevant features and target
X = train_data[['GrLivArea', 'BedroomAbvGr', 'TotalBath']]
y = train_data['SalePrice']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the linear regression model
model = LinearRegression()

# Fit the model
model.fit(X_train, y_train)

# Predicting the prices on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# To make predictions on the test dataset (test.csv), load it and use the same features
test_data = pd.read_csv('test.csv')
test_data['TotalBath'] = test_data['FullBath'] + (test_data['HalfBath'] * 0.5)

X_test_final = test_data[['GrLivArea', 'BedroomAbvGr', 'TotalBath']]
predictions = model.predict(X_test_final)

# Add predictions to the test DataFrame
test_data['PredictedPrice'] = predictions

# Save the predictions to a new CSV file
test_data[['Id', 'PredictedPrice']].to_csv('predictions.csv', index=False)