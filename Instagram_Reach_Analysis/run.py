from dataset_service import DatasetService
from model import PredictionModel
import numpy as np

if __name__ == "__main__":
    datasetService = DatasetService('./data/Instagram_data.csv')
    model = PredictionModel(datasetService)

    model.visualize()

    model.train()
    # Features = [['Likes','Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows']]
    features = np.array([[282.0, 233.0, 4.0, 9.0, 165.0, 54.0]])
    print(model.predict(features))

