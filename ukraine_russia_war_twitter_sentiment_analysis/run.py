from matplotlib import pyplot as plt
from wordcloud import STOPWORDS, WordCloud

from ukraine_russia_war_twitter_sentiment_analysis.data_service import DataService
from ukraine_russia_war_twitter_sentiment_analysis.model_service import ModelService

if __name__ == "__main__":
    dataService = DataService('data/ukraine_russia_war_twitter_sentiment_analysis.csv')
    # print(dataService.view(5))
    # print("application running")
    # print(dataService.data.columns)

    dataService.preprocessing()
    model = ModelService(dataService)
    model.load_algorithm()
    model.train()
    model.visualize()
