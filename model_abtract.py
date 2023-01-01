from abc import ABC, abstractmethod, abstractclassmethod


class ModelAbstract(ABC):

    def __init__(self, dataService):
        self.dataService = dataService
        self.model = None

    @abstractmethod
    def load_algorithm(self):
        pass

    def train(self):
        X_train, X_test, y_train, y_test = self.dataService.get_train_test_data()
        print("training started")
        self.model.fit(X_train, y_train)
        print('training complete')
        print(f"Accuracy of the trained model is: {self.model.score(X_test, y_test)}")
        return

    def predict(self, features):
        return self.model.predict(features)

    @abstractmethod
    def visualize(self):
        pass
