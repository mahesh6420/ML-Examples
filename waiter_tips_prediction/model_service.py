from sklearn.linear_model import LinearRegression

from model_abtract import ModelAbstract


class ModelService(ModelAbstract):

    def load_algorithm(self):
        self.model = LinearRegression()
        return

    def visualize(self):
        pass
