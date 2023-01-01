import nltk
from matplotlib import pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import STOPWORDS, WordCloud

from model_abtract import ModelAbstract


class ModelService(ModelAbstract):

    def load_algorithm(self):
        nltk.download('vader_lexicon')
        self.model = SentimentIntensityAnalyzer()

    def train(self):
        for tweet in self.dataService.data.tweet:
            self.dataService.data['Positive'] = self.model.polarity_scores(tweet)['pos']
            self.dataService.data['Negative'] = self.model.polarity_scores(tweet)['neg']
            self.dataService.data['Neutral'] = self.model.polarity_scores(tweet)['neu']

    def visualize(self):
        positive = ' '.join([i for i in self.dataService.data['tweet'][self.dataService.data['Positive'] > self.dataService.data["Negative"]]])
        negative = ' '.join([i for i in self.dataService.data['tweet'][self.dataService.data['Negative'] > self.dataService.data["Positive"]]])

        stopwords = set(STOPWORDS)
        wordcloud_pos = WordCloud(stopwords=stopwords, background_color="white").generate(positive)
        plt.figure(figsize=(15, 10))
        plt.imshow(wordcloud_pos, interpolation='bilinear')
        plt.axis("off")
        plt.show()

        # wordcloud_neg = WordCloud(stopwords=stopwords, background_color="white").generate(negative)
        # plt.figure(figsize=(15, 10))
        # plt.imshow(wordcloud_neg, interpolation='bilinear')
        # plt.axis("off")
        # plt.show()
