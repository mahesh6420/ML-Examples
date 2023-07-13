from Visualization.DataService import DataService
from Visualization.VisualizationService import VisualizationService

if __name__ == "__main__":
    # dataService = DataService("https://raw.githubusercontent.com/amankharwal/Website-data/master/tips.csv")
    dataService = DataService("data/onlinefraud.csv")
    dataService.preprocessing()

    visualization_service = VisualizationService(dataService)
    # visualization_service.visualize('bar', 'type')
    visualization_service.visualize_multiple('histogram')
    # visualization_service.visualize('histogram')