import pandas as pd

from waiter_tips_prediction.data_service import DataService
from waiter_tips_prediction.model_service import ModelService

if __name__ == "__main__":
    # dataService = DataService("https://raw.githubusercontent.com/amankharwal/Website-data/master/tips.csv")
    dataService = DataService("data/waiter_tips_data.csv")
    dataService.preprocessing()

    model = ModelService(dataService)
    model.load_algorithm()
    model.train()
    print(dataService.data.dtypes)
    # print(dataService.view(5))
    # model.train()
    # print(dataService.view(5))
