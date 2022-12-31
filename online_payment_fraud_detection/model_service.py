from sklearn.linear_model import PassiveAggressiveRegressor

from Instagram_Reach_Analysis.plot_service import Visualize
from model_abtract import ModelAbstract


class ModelService(ModelAbstract):
    def load_algorithm(self):
        self.model = PassiveAggressiveRegressor()

    def train(self):
        X_train, X_test, y_train, y_test = self.dataService.get_train_test_data(selected_columns=["type", "amount", "oldbalanceOrg", "newbalanceOrig"])
        print("training")
        self.model.fit(X_train, y_train)
        print('training complete')
        print(f"Accuracy of the trained model is: {self.model.score(X_test, y_test)}")
        return

    def visualize(self):
        plt = Visualize(self.dataService)
        plt.wordlcloud()

        return
