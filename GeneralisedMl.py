import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from sklearn import metrics
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from keras.utils.vis_utils import plot_model
import pickle
import pdb

class DataPipeline:
    def __init__(self):
        self.data = None
        self.features = None
        self.target = None
        self.model = None
        self.scaler = None
        self.history = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self, source, query=None, file_type='sqlite', **kwargs):
        """
        Load data from a specified source.

        Parameters:
        - source: path to the data source.
        - query: SQL query if loading from a database.
        - file_type: type of the data source ('sqlite', 'csv', etc.)
        """
        if file_type == 'sqlite':
            try:
                sqliteConnection = sqlite3.connect(source, timeout=20)
                if query is None:
                    raise ValueError("SQL query must be provided for sqlite database.")
                self.data = pd.read_sql_query(query, sqliteConnection)
                print("Data loaded successfully from SQLite database.")
            except sqlite3.Error as error:
                print("Failed to read data from sqlite table", error)
            finally:
                if sqliteConnection:
                    sqliteConnection.close()
                    print("The Sqlite connection is closed")
        elif file_type == 'csv':
            self.data = pd.read_csv(source, **kwargs)
            print("Data loaded successfully from CSV file.")
        else:
            raise ValueError("Unsupported file_type provided.")

    def preprocess_data(self, target_columns, feature_columns=None, balance_criteria=None):
        """
        Preprocess the data.

        Parameters:
        - target_columns: list of columns to be used as target variables.
        - feature_columns: list of columns to be used as features. If None, use all columns except target_columns.
        - balance_criteria: function or dict specifying criteria for balancing.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Please load data first.")

        # Handle missing values
        self.data.dropna(inplace=True)

        # Separate features and target
        self.target = self.data[target_columns]
        if feature_columns is None:
            feature_columns = list(set(self.data.columns) - set(target_columns))
        self.features = self.data[feature_columns]

        # Apply balancing if criteria provided
        if balance_criteria is not None:
            self.balance_data(balance_criteria)

    def balance_data(self, criteria):
        """
        Balance the dataset based on provided criteria.

        Parameters:
        - criteria: function or dict specifying criteria for balancing.
        """
        # Apply criteria to label the data
        self.data['label'] = None
        for label, func in criteria.items():
            self.data.loc[func(self.data), 'label'] = label

        # Remove rows without labels
        self.data.dropna(subset=['label'], inplace=True)

        # Now, balance the data
        counts = self.data['label'].value_counts()
        min_count = counts.min()
        balanced_data = self.data.groupby('label').apply(lambda x: x.sample(n=min_count, random_state=42))
        self.data = balanced_data.reset_index(drop=True)

        # Update features and target
        self.target = self.data['label']
        self.features = self.data.drop(columns=['label'])

    def split_data(self, test_size=0.1, random_state=101):
        """
        Split the data into training and testing sets.

        Parameters:
        - test_size: proportion of the dataset to include in the test split.
        - random_state: controls the shuffling applied before the split.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features, self.target, test_size=test_size, stratify=self.target, random_state=random_state)

    def scale_data(self):
        """
        Scale the features using StandardScaler.
        """
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        print("Data scaling complete.")

    def build_model(self, input_shape, layers_config=None):
        """
        Build a neural network model.

        Parameters:
        - input_shape: shape of the input features.
        - layers_config: list of dicts specifying the layers. If None, a default configuration is used.
        """
        if layers_config is None:
            layers_config = [
                {'units': input_shape, 'activation': 'relu', 'dropout': 0.5, 'batch_norm': True},
                {'units': input_shape//2, 'activation': 'relu', 'dropout': 0.5, 'batch_norm': True},
                {'units': input_shape//4, 'activation': 'relu', 'dropout': 0.5, 'batch_norm': True},
            ]

        self.model = Sequential()
        for layer in layers_config:
            self.model.add(Dense(layer['units'], activation=layer['activation'], kernel_regularizer=l2(0.01)))
            if layer.get('batch_norm', False):
                self.model.add(BatchNormalization())
            if layer.get('dropout', 0) > 0:
                self.model.add(Dropout(layer['dropout']))
        # Output layer
        self.model.add(Dense(1, activation='sigmoid'))
        print("Model built successfully.")

    def train_model(self, epochs=100, batch_size=32, patience=10):
        """
        Train the model.

        Parameters:
        - epochs: number of epochs to train.
        - batch_size: size of the batches.
        - patience: number of epochs with no improvement after which training will be stopped.
        """
        if self.model is None:
            raise ValueError("Model not built. Please build the model first.")

        # Compile the model
        optimiser = Adam(learning_rate=0.001)
        self.model.compile(loss='binary_crossentropy', optimizer=optimiser, metrics=['accuracy'])

        # Early stopping
        early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)

        # Learning rate scheduler
        def scheduler(epoch, lr):
            if epoch < 10:
                return lr
            else:
                return lr * tf.math.exp(-0.1)
        lr_schedule = LearningRateScheduler(scheduler)

        # Train the model
        self.history = self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size,
                                      validation_data=(self.X_test, self.y_test),
                                      callbacks=[early_stop, lr_schedule])
        print("Model training complete.")

    def evaluate_model(self):
        """
        Evaluate the model on the test set.
        """
        if self.model is None:
            raise ValueError("Model not built. Please build the model first.")

        predictions = self.model.predict(self.X_test)
        predictions = np.round(predictions).astype(int)
        print("Classification Report:")
        print(classification_report(self.y_test, predictions))
        print("Confusion Matrix:")
        print(confusion_matrix(self.y_test, predictions))

    def save_model(self, model_path, scaler_path):
        """
        Save the model and the scaler.

        Parameters:
        - model_path: path to save the model.
        - scaler_path: path to save the scaler.
        """
        self.model.save(model_path)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")

    def plot_model(self, plot_path):
        """
        Plot the model architecture.

        Parameters:
        - plot_path: path to save the plot.
        """
        plot_model(self.model, to_file=plot_path, show_shapes=True, show_layer_names=True)
        print(f"Model architecture plot saved to {plot_path}")
