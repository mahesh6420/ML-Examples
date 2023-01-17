import numpy as np

from data_abstract import DataAbstract


class DataService(DataAbstract):

    def __init__(self, data):
        self.data = data
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def preprocessing(self):
        self.data['Date'] = self.data.index
        self.data.reset_index(drop=True, inplace=True)

        self.data = self.data[['Open', 'High', 'Volume', 'Low', 'Close']]
        # self.data = self.data[columns]

        self.get_train_test_data()

