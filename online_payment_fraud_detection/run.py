from online_payment_fraud_detection.dataset_service import DatasetService
from online_payment_fraud_detection.model_service import ModelService
import numpy as np


if __name__ == "__main__":
    datasetService = DatasetService('./data/online_payment_fraud_data.csv')

    # print(datasetService.view(5))
    # print(datasetService.data.isnull().sum())

    #    correlation of data with the prediction column - isFraud in this case
    correlation = datasetService.data.corr()
    # print(correlation['isFraud'].sort_values(ascending=False))

    datasetService.preprocessing()
    # print(datasetService.data['type'])

    model = ModelService(datasetService)
    model.train()

    # prediction
    # features = [type, amount, oldbalanceOrg, newbalanceOrig]
    features = np.array([[4, 9000.60, 9000.60, 0.0]])
    print(model.predict(features))
