from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS


class Visualize:
    def __init__(self, dataService):
        self.datasetService = dataService

    def wordlcloud(self):
        text = " ".join(i for i in self.datasetService.data.Caption)
        stopwords = set(STOPWORDS)
        wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
        plt.style.use('classic')
        plt.figure(figsize=(12, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()

        return
