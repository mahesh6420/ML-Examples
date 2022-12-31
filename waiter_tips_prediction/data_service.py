import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data_abstract import DataAbstract


class DataService(DataAbstract):
    """
    Class that has methods that works with a data.
    like view, get_columns, preprocessing etc.
    """

    def preprocessing(self):
        self.convert_to_category(['sex', 'smoker', 'day', 'time'])

        # TODO: automate this
        self.data["sex"] = self.data["sex"].map({"Female": 0, "Male": 1})
        self.data["smoker"] = self.data["smoker"].map({"No": 0, "Yes": 1})
        self.data["day"] = self.data["day"].map({"Thur": 0, "Fri": 1, "Sat": 2, "Sun": 3})
        self.data["time"] = self.data["time"].map({"Lunch": 0, "Dinner": 1})


