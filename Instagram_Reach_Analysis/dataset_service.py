import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DatasetService:
    def __init__(self, link):
        self.data = pd.read_csv(link)

    # def load(self, link):
    #
    #     return self.data

    def get_columns(self):
        return self.data.columns

    def preprocessing(self):
        self.data.dropna(inplace=True)

    def get_train_test_data(self):
        x = np.array(self.data[['Likes', 'Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows']])
        y = np.array(self.data["Impressions"])
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test

