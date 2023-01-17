from datetime import date, timedelta

import numpy as np
import yfinance as yf

from stock_price_prediction.data_service import DataService
from stock_price_prediction.model_service import ModelService

if __name__ == "__main__":
    today = date.today()
    d1 = today.strftime("%Y-%m-%d")
    end_date = d1

    d2 = today - timedelta(days=5000)
    d2 = d2.strftime("%Y-%m-%d")
    start_date = d2

    data = yf.download("AAPL",
                       start=start_date,
                       end=end_date,
                       progress=False)
    # print(type(data))
    dataService = DataService(data)
    dataService.preprocessing()

    model = ModelService(dataService)
    model.load_algorithm()
    model.train()

    # features = [Open, High, Low, Adj Close, Volume]
    features = np.array([[177.089996, 180.419998, 177.070007, 74919600]])
    print(model.predict(features))

