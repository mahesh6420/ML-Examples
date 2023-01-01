from data_abstract import DataAbstract
import plotly.express as px
import plotly.graph_objects as go

import plotly.io as pio

class DataService(DataAbstract):

    def preprocessing(self):
        pass

    def visualize_scatter_plot(self, x, y):
        pio.renderers.default = "browser"
        """
        scatter plot between two features x and y
        :param x:
        :param y:
        :return:
        """
        figure = px.scatter(data_frame=self.data, x=x,
                            y=y, size=y, trendline="ols")
        figure.show()
