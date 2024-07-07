import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
data = pd.read_csv('electric_car.csv')

# Display the first few rows of the dataframe to inspect the columns
print(data.head())
print(data.columns)

# Display data types of columns
print(data.dtypes)

# Select relevant numeric features and target variable
numeric_columns = ['City (kWh/100 km)', 'Highway (kWh/100 km)', 'Combined (kWh/100 km)', 'CO2 emissions (g/km)']
data = data[numeric_columns]

# Drop rows with missing values in these columns
data = data.dropna()

# Separate features and target
features = data[['City (kWh/100 km)', 'Highway (kWh/100 km)', 'Combined (kWh/100 km)']]
target = data['CO2 emissions (g/km)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))  # Output layer for regression

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, batch_size=10)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')

# Make predictions
y_pred = model.predict(X_test)

# Display first few predictions
print(y_pred[:5])

