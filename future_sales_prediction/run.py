import numpy as np

from future_sales_prediction.data_service import DataService
from future_sales_prediction.model_service import ModelService

if __name__ == "__main__":
    dataService = DataService('data/future_sales_prediction_data.csv')
    print(dataService.check_null())
    # dataService.visualize_scatter_plot(x="Sales", y="TV")
    # dataService.visualize_scatter_plot(x="Sales", y="Newspaper")
    # dataService.visualize_scatter_plot(x="Sales", y="Radio")

    dataService.get_correlation('Sales')

    model = ModelService(dataService)
    model.load_algorithm()
    model.train()

    features = np.array([[230.1, 37.8, 69.2]])
    print(model.predict(features))
