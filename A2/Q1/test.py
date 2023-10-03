from collections import defaultdict
import pandas
import numpy as np

NEUTRAL = 0
POSITIVE = 1
NEGATIVE = 2
LABELS = ['Neutral', 'Positive', 'Negative']

def read_data(path: str):
    reader = pandas.read_csv(path)
    reader['CoronaTweet'] = reader['CoronaTweet'].str.lower()
    return reader

# # Returns an array
# # P(c)
def prior(df):
    c0, c1, c2 = df[df.Sentiment == 'Neutral'].shape[0], df[df.Sentiment == 'Positive'].shape[0], df[df.Sentiment == 'Negative'].shape[0]
    ndoc = df.shape[0]
    ret = np.array([np.log(c0/ndoc), np.log(c1/ndoc), np.log(c2/ndoc)])
    return ret

# log P(w | c)
# Returns a matrix
def loglikelihood(df, V):
    llhc = defaultdict(lambda: 0)
    for i, c in enumerate(LABELS):
        count = defaultdict(lambda: 0)
        total = 0
        for tweets in df[df.Sentiment == c].CoronaTweet:
            tweet = set(tweets.split())
            for word in tweet:
                count[word] += 1
                total += 1

        laplace = total + len(V)
        for word in V:
            llhc[(word, i)] = np.log((count[word] + 1) / (total + laplace))

        print(count['i'] + 1, total, len(V))
        # print(np.log(2137/(411236 + 113583)))
        print(llhc[('i', 2)])

    return llhc

def vocabulary(df):
    V = set()
    for tweets in df.CoronaTweet:
        tweet = tweets.split()
        for word in tweet:
            V.add(word)
    return V

def test(text, LPC, LLWC, V):
    text = set(text.split())
    mx = -1e18
    ret = -1
    for i, c in enumerate(LABELS):
        sum = LPC[i]
        for word in text:
            if (word in V):
                sum += LLWC[(word, i)]
        if (sum > mx):
            mx = sum
            ret = c
    return ret

df = read_data('Q1/Corona_train.csv')
LPC = prior(df)
print(LPC)
V = vocabulary(df)
LLWC = loglikelihood(df, V)
print(LLWC[('i', 2)])

test_data = read_data('Q1/Corona_validation.csv')
# print(test_data)

cnt = 0
for i in range(test_data.shape[0]):
    cnt += 1 if test(test_data.CoronaTweet[i], LPC, LLWC, V) == test_data.Sentiment[i] else 0

print(cnt/test_data.shape[0], cnt)
