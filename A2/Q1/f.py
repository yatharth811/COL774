# https://web.stanford.edu/~jurafsky/slp3/4.pdf
import pandas as pd
import numpy as np
from collections import defaultdict
import regex as re
from utils import read_data, lower
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

NEUTRAL = 'Neutral'
POSITIVE = 'Positive'
NEGATIVE = 'Negative'
# stopwords = set(stopwords.words('english'))
# print(stopwords)
ps = PorterStemmer()
# print(stopwords)

class NaiveBayes():
    def __init__(self):
        self.vocabulary = set()
        self.priors = {}
        self.conditionals = {}
        self.wordcloud = {}
        pass
    
    def preprocess(self, text):
        # This leads to 71% accuracy.
        # return [x.strip(r'[. "\'#?&,;]') for x in re.split(r'[\s\n,.]+', text.lower())]
        # return [ps.stem(x) for x in text.strip().lower().split()]
        # return [ps.stem(x) for x in text.lower().strip().split()]
        return [ps.stem(word) for word in text.strip().split()]
        # return [ps.stem(x.strip(r'[. "\'#?&,;]')) for x in set(re.split(r'[\s\n,.]+', text.lower()))]
    
    def train(self, training_data):
        word_counts = defaultdict(lambda: {NEUTRAL: 0, POSITIVE: 0, NEGATIVE: 0})
        class_counts = {NEUTRAL: 0, POSITIVE: 0, NEGATIVE: 0}
        prior_counts = {NEUTRAL: 0, POSITIVE: 0, NEGATIVE: 0}
        cloud_counts = defaultdict(lambda: defaultdict(int))
        total_text = training_data.shape[0]
        
        for i in range(total_text):
            sentiment, text = training_data.Sentiment[i], training_data.CoronaTweet[i]
            for word in self.preprocess(text):
                # if (bool(re.search(r'http.', word, flags=re.IGNORECASE))):
                #     continue
                # if word in stopwords:
                #   continue
              
                word_counts[word][sentiment] += 1
                cloud_counts[sentiment][word] += 1
                class_counts[sentiment] += 1
                self.vocabulary.add(word)
            prior_counts[sentiment] += 1
                
        # Prior Class Probability
        priors = {}
        for sentiment, count in prior_counts.items():
            priors[sentiment] = np.log(count / total_text)
        # print(priors, count, total_text)

        # Conditional Probability
        conditionals = {}
        for word in self.vocabulary:
            conditionals[word] = {}
            for sentiment in [NEUTRAL, POSITIVE, NEGATIVE]:
                # Laplace smoothing (add-one smoothing)
                count = word_counts[word][sentiment] + 1
                total_words_in_class = class_counts[sentiment] + len(self.vocabulary)
                conditionals[word][sentiment] = np.log(count / total_words_in_class)

        self.priors, self.conditionals, self.wordcloud = priors, conditionals, cloud_counts
    
    def classify(self, text):
        scores = {NEUTRAL: 0, POSITIVE: 0, NEGATIVE: 0}
        words = self.preprocess(text)
        
        for sentiment in [NEUTRAL, POSITIVE, NEGATIVE]:
            scores[sentiment] = self.priors[sentiment]
            for word in words:
                if word in self.vocabulary:
                    scores[sentiment] += self.conditionals[word][sentiment]
        
        return max(scores, key=scores.get)
    
    def test(self, test_data):
        cnt = 0
        for i in range(test_data.shape[0]):
            cnt += 1 if model.classify(test_data.CoronaTweet[i]) == test_data.Sentiment[i] else 0
        return cnt / test_data.shape[0]
    
    def get_word_cloud(self):
        for sentiment in [NEUTRAL, POSITIVE, NEGATIVE]:
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(self.wordcloud[sentiment])
            # Display the word cloud using Matplotlib
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.show()
            


dirs = [1, 2, 5, 10, 25, 50, 100]

for path in dirs:
  model = NaiveBayes()
  df1 = pd.read_csv('Q1/Corona_train.csv')
  df2 = pd.read_csv(f'Q1/Domain_Adaptation/Twitter_train_{path}.csv')
  df2 = df2.rename(columns={'Tweet': 'CoronaTweet'})
  concatenated_df = pd.concat([df1, df2], ignore_index=True)
  model.train(concatenated_df)
  test_data = read_data('Q1/Domain_Adaptation/Twitter_validation.csv')
  test_data = test_data.rename(columns={'Tweet': 'CoronaTweet'})
  # test_data = lower(test_data)
  # test_data['CoronaTweet'] = test_data['CoronaTweet'].str.lower()
  print(f"Accuracy with corona and {path}% twitter training data: ", model.test(test_data))
  # model.get_word_cloud()
  

for path in dirs:
  model = NaiveBayes()
  df2 = pd.read_csv(f'Q1/Domain_Adaptation/Twitter_train_{path}.csv')
  df2 = df2.rename(columns={'Tweet': 'CoronaTweet'})
  model.train(df2)
  test_data = read_data('Q1/Domain_Adaptation/Twitter_validation.csv')
  test_data = test_data.rename(columns={'Tweet': 'CoronaTweet'})
  # test_data = lower(test_data)
  # test_data['CoronaTweet'] = test_data['CoronaTweet'].str.lower()
  print(f"Accuracy with only {path}% twitter training data: ", model.test(test_data))
  # model.get_word_cloud()