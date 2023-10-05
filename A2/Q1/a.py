# https://web.stanford.edu/~jurafsky/slp3/4.pdf
import pandas as pd
import numpy as np
from collections import defaultdict
import regex as re
from utils import read_data, lower
from sklearn.metrics import confusion_matrix
from wordcloud import WordCloud
import matplotlib.pyplot as plt

NEUTRAL = 'Neutral'
POSITIVE = 'Positive'
NEGATIVE = 'Negative'

class NaiveBayes():
    
    def __init__(self):
        self.vocabulary = set()
        self.priors = {}
        self.conditionals = {}
        self.wordcloud = {}
        pass
    
    def preprocess(self, text):
        # This leads to 71% accuracy.
        # return [x.strip(r'[. "\'#?&,;]') for x in (re.split(r'[\s\n,.]+', text.lower()))]
        # return re.split(r'[\s\n,.]+', text)
        return text.strip().split()
    
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

    def get_predictions(self, test_data):
        predictions = []
        for i in range(test_data.shape[0]):
            predictions.append(model.classify(test_data.CoronaTweet[i]) )
        return predictions
    
    def get_word_cloud(self):
         for sentiment in [NEUTRAL, POSITIVE, NEGATIVE]:
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(self.wordcloud[sentiment])
            # Display the word cloud using Matplotlib
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.show()

model = NaiveBayes()
model.train(pd.read_csv('Q1/Corona_train.csv'))
test_data = read_data('Q1/Corona_train.csv')
# lower(test_data)
print("Accuracy with training data: ", model.test(test_data))
print("Confusion Matrix:\n", confusion_matrix(np.array(test_data['Sentiment']), model.get_predictions(test_data)))
model.get_word_cloud()

test_data = read_data('Q1/Corona_validation.csv')
# lower(test_data)
print("Accuracy with validation data: ", model.test(test_data))
print("Confusion Matrix:\n", confusion_matrix(np.array(test_data['Sentiment']), model.get_predictions(test_data)))
model.get_word_cloud()