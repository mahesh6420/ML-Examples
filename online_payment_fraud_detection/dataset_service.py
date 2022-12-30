import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DatasetService:
    def __init__(self, link):
        self.data = pd.read_csv(link)

    # def load(self, link):
    #
    #     return self.data

    def view(self, row_numbers):
        return self.data.head(row_numbers)

    def get_columns(self):
        return self.data.columns

    def preprocessing(self):
        #     transforming categorical features to numerical
        self.data['type'] = self.data['type'].map({
            "CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3,
            "TRANSFER": 4, "DEBIT": 5
        })

    def get_train_test_data(self):
        x = np.array(self.data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
        y = np.array(self.data[["isFraud"]])
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test
