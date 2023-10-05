# https://web.stanford.edu/~jurafsky/slp3/4.pdf
import pandas as pd
import numpy as np
from collections import defaultdict
import regex as re
from utils import read_data, lower
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

NEUTRAL = 'Neutral'
POSITIVE = 'Positive'
NEGATIVE = 'Negative'
stopwords = set(stopwords.words('english'))
ps = PorterStemmer()

class NaiveBayes():
    def __init__(self):
        self.vocabulary = set()
        self.priors = {}
        self.conditionals = {}
        self.doc_lengths = {NEUTRAL: 0, POSITIVE: 0, NEGATIVE: 0}  # To store document lengths
        self.doc_term_counts = defaultdict(lambda: defaultdict(int))  # To store term frequencies
        pass
    
    def preprocess(self, text):
        # This leads to 71% accuracy.
        return [ps.stem(x.strip(r'[. "\'#?&,;]')) for x in re.split(r'[\s\n,.]+', text.lower()) if x not in stopwords]
        # return [ps.stem(x) for x in text.strip().lower().split() if x not in stopwords]
        # return [ps.stem(x) for x in text.lower().strip().split()]
        # return ([ps.stem(word) for word in text.strip().split()])
        # return [ps.stem(x.strip(r'[. "\'#?&,;]')) for x in set(re.split(r'[\s\n,.]+', text.lower()))]
        
    def construct_bigrams(self, preprocessed_text):
      size = len(preprocessed_text)
      # preprocessed_text = list(preprocessed_text)
      # bigrams = set()
      bigrams = []
      
      for i in range(0, size - 1):
        bigrams.append(preprocessed_text[i] + preprocessed_text[i + 1])
      bigrams = preprocessed_text + bigrams
      return bigrams
    
    def calculate_tfidf(self, word, sentiment):
        # Calculate Term Frequency (TF)
        tf = self.doc_term_counts[sentiment][word] / self.doc_lengths[sentiment]
        
        # Calculate Inverse Document Frequency (IDF)
        doc_count_with_term = sum(1 for s in [NEUTRAL, POSITIVE, NEGATIVE] if word in self.doc_term_counts[s])
        idf = np.log((3 + 1) / (doc_count_with_term + 1)) + 1  # Laplace smoothed IDF
        
        # Calculate TF-IDF
        tfidf = tf * idf
        return tfidf
    
    def train(self, training_data):
        word_counts = defaultdict(lambda: {NEUTRAL: 0, POSITIVE: 0, NEGATIVE: 0})
        class_counts = {NEUTRAL: 0, POSITIVE: 0, NEGATIVE: 0}
        prior_counts = {NEUTRAL: 0, POSITIVE: 0, NEGATIVE: 0}
        total_text = training_data.shape[0]
        
        for i in range(total_text):
            sentiment, text = training_data.Sentiment[i], training_data.CoronaTweet[i]
            words = self.preprocess(text)
            bigrams = self.construct_bigrams(words)
            self.doc_lengths[sentiment] += len(bigrams)
            for word in bigrams:
                # if (bool(re.search(r'http.', word, flags=re.IGNORECASE))):
                #     continue
                # if word in stopwords:
                #   continue  
                word_counts[word][sentiment] += 1
                class_counts[sentiment] += 1
                self.vocabulary.add(word)
                self.doc_term_counts[sentiment][word] += 1
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

        self.priors, self.conditionals = priors, conditionals
    
    def classify(self, text):
        scores = {NEUTRAL: 0, POSITIVE: 0, NEGATIVE: 0}
        words = self.preprocess(text)
        bigrams = self.construct_bigrams(words)
        
        for sentiment in [NEUTRAL, POSITIVE, NEGATIVE]:
            scores[sentiment] = self.priors[sentiment]
            for word in bigrams:
              if word in self.vocabulary:
                # print(self.calculate_tfidf(word, sentiment))
                # scores[sentiment] += self.conditionals[word][sentiment] + 200 * self.calculate_tfidf(word, sentiment)
                scores[sentiment] += self.conditionals[word][sentiment]
        return max(scores, key=scores.get)
    
    def test(self, test_data):
        cnt = 0
        for i in range(test_data.shape[0]):
            cnt += 1 if model.classify(test_data.CoronaTweet[i]) == test_data.Sentiment[i] else 0
        return cnt / test_data.shape[0]
        

model = NaiveBayes()
model.train(pd.read_csv('Q1/Corona_train.csv'))
test_data = read_data('Q1/Corona_train.csv')
print("Accuracy on training data: ", model.test(test_data) * 100)
test_data = read_data('Q1/Corona_validation.csv')
print("Accuracy on validation data: ", model.test(test_data) * 100)