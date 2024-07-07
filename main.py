import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


class ElectricCarEmissionModel:
    """
    A class used to represent an Electric Car Emission Model

    ...

    Attributes
    ----------
    file_path : str
        a formatted string to define the path of csv file
    data : DataFrame
        a pandas DataFrame to store the dataset
    X_train : DataFrame
        a pandas DataFrame for the input training set
    X_test : DataFrame
        a pandas DataFrame for the input testing set
    y_train : DataFrame/Series
        a pandas DataFrame or Series for the target training set
    y_test : DataFrame/Series
        a pandas DataFrame or Series for the target testing set
    model : Object
        a Sequential object from keras

    Methods
    -------
    load_data():
        Loads the data from csv file and preprocess it (like drops NaN values).

    split_data():
        Splits the loaded data into input and target sets and also split them into training and testing sets.

    normalize_data():
        Function to normalize the training and test input data.

    build_model():
        Function to initialize a Sequential model.

    compile_model():
        Function to compile the defined model.

    train_model():
        Function to train the model using the training set data.

    evaluate_model():
        Function to evaluate the model performance using test set data.

    predict():
        Function to predict target using the test input data.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

    def load_data(self):
        """Loads and pre-processes the data from the file path."""
        self.data = pd.read_csv(self.file_path)
        numeric_columns = ['City (kWh/100 km)',
                           'Highway (kWh/100 km)',
                           'Combined (kWh/100 km)',
                           'CO2 emissions (g/km)']
        self.data = self.data[numeric_columns]
        self.data = self.data.dropna()

    def split_data(self):
        """Splits the data into input and target datasets for training and testing."""
        features = self.data[['City (kWh/100 km)',
                              'Highway (kWh/100 km)',
                              'Combined (kWh/100 km)']]
        target = self.data['CO2 emissions (g/km)']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            features, target, test_size=0.2, random_state=42)

    def normalize_data(self):
        """Normalizes the input data using StandardScaler."""
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def build_model(self):
        """Builds a Sequential model with two dense layers and one output layer."""
        self.model = Sequential()
        self.model.add(Dense(64, input_dim=self.X_train.shape[1],
                             activation='relu'))  # Input layer
        self.model.add(Dense(32, activation='relu'))  # Hidden layer
        self.model.add(Dense(1))  # Output layer for regression

    def compile_model(self):
        """Compiles the model with 'adam' optimizer and 'mean_squared_error' as loss function."""
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train_model(self):
        """Trains the model on the training data."""
        self.model.fit(self.X_train, self.y_train,
                       epochs=100,
                       validation_split=0.2,
                       batch_size=10)

    def evaluate_model(self):
        """Evaluates the model's performance using the test set data."""
        loss = self.model.evaluate(self.X_test, self.y_test)
        print(f'Test loss: {loss}')

    def predict(self):
        """Makes predictions using the test input data."""
        y_pred = self.model.predict(self.X_test)
        print(y_pred[:5])


if __name__ == "__main__":
    model = ElectricCarEmissionModel('electric_car.csv')
    model.load_data()
    model.split_data()
    model.normalize_data()
    model.build_model()
    model.compile_model()
    model.train_model()
    model.evaluate_model()
    model.predict()
