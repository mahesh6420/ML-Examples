from keras import Sequential
from keras.layers import LSTM, Dense

from model_abtract import ModelAbstract
import plotly.graph_objects as go


class ModelService(ModelAbstract):

    def load_algorithm(self):
        self.model = Sequential()

        self.model.add(LSTM(128, return_sequences=True, input_shape=(self.dataService.X_train.shape[1], 1)))
        self.model.add(LSTM(64, return_sequences=True))
        self.model.add(Dense(25))
        self.model.add(Dense(1))

        print(self.model.summary())

        return self.model

    def train(self):
        self.model.compile(optimizer='adam', loss='mean_squared_error')

        print("training started")
        self.model.fit(self.dataService.X_train, self.dataService.y_train, batch_size=1, epochs=30)
        print('training complete')
        # print(f"Accuracy of the trained model is: {self.model..score(self.dataService.X_test, self.dataService.y_test)}")

        return

    def visualize(self):
        figure = go.Figure(data=[go.Candlestick(x=self.dataService.data["Date"],
                                                open=self.dataService.data["Open"],
                                                high=self.dataService.data["High"],
                                                low=self.dataService.data["Low"],
                                                close=self.dataService.data["Close"])])
        figure.update_layout(title="Apple Stock Price Analysis",
                             xaxis_rangeslider_visible=False)
        figure.show()
