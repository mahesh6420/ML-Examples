from abc import ABC, abstractmethod, abstractclassmethod


class ModelAbstract(ABC):

    def __init__(self, dataService):
        self.dataService = dataService
        self.model = None

    @abstractmethod
    def load_algorithm(self):
        pass

    def train(self):
        print("training started")
        self.model.fit(self.dataService.X_train, self.dataService.y_train)
        print('training complete')
        print(f"Accuracy of the trained model is: {self.model.score(self.dataService.X_test, self.dataService.y_test)}")
        return

    def predict(self, features):
        return self.model.predict(features)

    @abstractmethod
    def visualize(self):
        pass
