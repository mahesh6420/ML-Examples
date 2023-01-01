import re
import string

import nltk
from nltk.corpus import stopwords

from data_abstract import DataAbstract


class DataService(DataAbstract):

    def preprocessing(self):
        # We only need three columns for this task (username, tweet, and language)
        self.data = self.data[['username', 'tweet', 'language']]

        self.data['tweet'] = self.data['tweet'].apply(self.clean)

    def clean(self, text):
        """
        will remove all the links, punctuation, symbols and other language errors from the tweets
        :text
        """
        nltk.download('stopwords')
        stemmer = nltk.SnowballStemmer('english')
        stopword = set(stopwords.words('english'))

        text = str(text).lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        text = [word for word in text.split(' ') if word not in stopword]
        text = " ".join(text)
        text = [stemmer.stem(word) for word in text.split(' ')]
        text = " ".join(text)

        return text
