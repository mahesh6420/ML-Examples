from sklearn.linear_model import PassiveAggressiveRegressor

from Instagram_Reach_Analysis.plot_service import Visualize


class ModelService:
    def __init__(self, datasetService):
        self.datasetService = datasetService
        self.model = None

    def load_algorithm(self):
        self.model = PassiveAggressiveRegressor()

    def train(self):
        self.load_algorithm()
        X_train, X_test, y_train, y_test = self.datasetService.get_train_test_data()
        self.model.fit(X_train, y_train)
        print(self.model.score(X_test, y_test))

    def predict(self, features):
        return self.model.predict(features)

    def visualize(self):
        plt = Visualize(self.datasetService)
        plt.wordlcloud()

        return
