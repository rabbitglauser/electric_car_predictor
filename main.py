import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
data = pd.read_csv('electric_car.csv')

# Assuming the CSV has headers; if not, add headers accordingly
data.columns = [
    'Year', 'Make', 'Model', 'Vehicle Class', 'Fuel Consumption City (L/100km)',
    'Fuel Consumption Hwy (L/100km)', 'Fuel Consumption Comb (L/100km)',
    'Fuel Consumption Comb (mpg)', 'CO2 Emissions (g/km)', 'CO2 Rating', 'Smog Rating'
]

# Select relevant features and target variable
features = data[['Fuel Consumption City (L/100km)', 'Fuel Consumption Hwy (L/100km)', 'Fuel Consumption Comb (L/100km)']]
target = data['CO2 Emissions (g/km)']

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
