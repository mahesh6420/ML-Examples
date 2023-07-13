import matplotlib.pyplot as plt


class VisualizationService:

    def __init__(self, data_service):
        self.dataService = data_service

    def visualize(self, type_of_plot, x_axis, y_axis=None):

        if type_of_plot == 'box':
            self.boxplot(x_axis)
        elif type_of_plot == 'histogram':
            self.histogram(x_axis)
        elif type_of_plot == 'scatter':
            self.scatter(x_axis, y_axis)
        elif type_of_plot == 'bar':
            self.bar(x_axis, y_axis)
        else:
            print('Invalid visualization type')

    def visualize_multiple(self, type_of_plot):
        for column in self.dataService.data.columns:
            if type_of_plot == 'histogram':
                self.histogram(column)
            elif type_of_plot == 'box':
                self.boxplot(column)
            elif type_of_plot == 'scatter':
                for column2 in self.dataService.data.columns:
                    if column != column2:
                        self.scatter(column, column2)
            elif type_of_plot == 'bar':
                for column2 in self.dataService.data.columns:
                    if column != column2:
                        self.bar(column, column2)
            else:
                print('Invalid visualization type')

    def histogram(self, column):
        self.dataService.data[column].value_counts().plot(kind='hist')
        plt.show()

        return

    def boxplot(self, column):
        self.dataService.data[column].plot(kind='box')
        plt.show()

        return

    def scatter(self, x_axis, y_axis):
        self.dataService.data.plot(kind='scatter', x=x_axis, y=y_axis)
        plt.show()

        return

    def bar(self, x_axis, y_axis):
        self.dataService.data.plot(kind='bar', x=x_axis, y=y_axis)
        plt.show()

        return

