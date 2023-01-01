from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

class DataAbstract(ABC):

    def __init__(self, dataUrl):
        """
        Initiate class with url of the data
        :param dataUrl: url of the data
        """
        self.data = pd.read_csv(dataUrl)

    def view(self, row_numbers):
        """
        Shows top 5 rows of the data if row_numbers is 5
        :param row_numbers: numbers of row to show
        :return:
        """
        return self.data.head(row_numbers)

    def get_columns(self):
        """
        :return: list of column names
        """
        return self.data.columns

    def get_train_test_data(self, selected_columns=None):
        """
        make sure (in the data) the last column is the feature you want to predict/classify
        :selected_columns: pass a list of columns to consider for the training process
        :return: X_train, X_test, y_train, y_test
        """
        columns = self.get_columns()
        print(columns)
        print(columns[:-1])
        print(columns[-1])

        print(selected_columns)
        x = np.array(self.data[columns[:-1] if selected_columns is None else selected_columns])
        y = np.array(self.data[[columns[-1]]])

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test

    def convert_to_category(self, list_of_column_names):
        for column_name in list_of_column_names:
            self.data[f'{column_name}'] = pd.Categorical(self.data[f'{column_name}'])

        return

    def check_null(self):
        print(self.data.isnull().sum())

    def get_correlation(self, column_name):
        correlation = self.data.corr()
        print(f'Correlation of {column_name} with other columns is: ')
        print(correlation[f'{column_name}'].sort_values(ascending=False))

    @abstractmethod
    def preprocessing(self):
        pass
