import numpy as np
import csv
from sklearn.metrics import f1_score
from collections import *
from string import punctuation
import re
import decimal
import plotly as py
import plotly.graph_objs as go
import webbrowser

###  Part 1: Naive Bayes Classifier
def opening(fname):
    rdata = []
    f = open(fname, 'r', encoding = 'utf-8')
    for x in f.readlines()[1:]:
        rdata.append(re.split('\t', x.replace('\n', '')))
    return rdata

def tokenize(string):
    return string.split()

def better_tokenize(string):
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    no_punct = ""
    for char in string:
        if char not in punctuations:
            no_punct = no_punct + char.lower()
    ls_words = no_punct.split()
    common_words = open('./words/common-words.txt').read()
    token_ls = [word for word in ls_words if word not in common_words]
    return token_ls


class TSV():
    """Organizing each row in the tsv file"""
    def __init__(self, row):
        self.ID = row[0]
        self.tweet = better_tokenize(row[1])
        try:
            self.classifer = row[2]
        except:
            pass

def create_instances(row):
    return [TSV(x) for x in row]

def train(inst):
    notHate_dict = {}
    hate_dict = {}
    tot = 0
    tot_hate_messages = 0
    n_NotHate = 0
    n_Hate = 0
    for row in inst:
        if row.classifer == '1':
            tot_hate_messages += 1
            for word in row.tweet:
                n_Hate += 1
                hate_dict[word] = hate_dict.get(word, 0) + 1
        else:
            for word in row.tweet:
                n_NotHate += 1
                notHate_dict[word] = notHate_dict.get(word, 0) + 1
        tot += 1
    pH = (tot_hate_messages)/tot
    pNotH = (tot - tot_hate_messages)/tot
    return n_Hate, n_NotHate, tot, pH, pNotH, notHate_dict, hate_dict

def conditionalWord(word, hate, smoothing_alpha):
    if hate:
        return (hate_dict.get(word,0)+smoothing_alpha)/(n_Hate+smoothing_alpha*tot)
    return  (notHate_dict.get(word, 0)+smoothing_alpha)/(n_NotHate+smoothing_alpha*tot)

def conditionalTweet(tweet, hate, smoothing_alpha):
    result = 1.0
    for word in tweet:
        result *= conditionalWord(word, hate, smoothing_alpha)
    return result

def classify(tweet, pH, pNotH, smoothing_alpha, testing = False):
    isHate = pH * conditionalTweet(tweet.tweet, True, smoothing_alpha)
    isNotHate = pNotH * conditionalTweet(tweet.tweet, False, smoothing_alpha)
    if isHate > isNotHate:
        cs = '1'
        if not testing:
            return (tweet.ID, cs)
    else:
        cs = '0'
        if not testing:
            return (tweet.ID, cs)
    if testing:
        if cs == tweet.classifer:
            result = 'match'
        else:
            result = 'not match'
        return tweet.ID, cs, result

###  Training the LM
o = opening('train.tsv')
z = create_instances(o)

n_Hate, n_NotHate, tot, pH, pNotH, notHate_dict, hate_dict = train(z)

o2 = opening('test.unlabeled.tsv')
z2 = create_instances(o2)

#  Testing on DEV data
# o2 = opening('dev.tsv')
# z2 = create_instances(o2)

# y_true = []
# y_pred = []
full_array = []
for x2 in z2:
    x = np.array(classify(x2, pH, pNotH, smoothing_alpha = 0.9, testing = False))
    full_array.append(x)
    print(x)

### Writing the out.csv file that is used to upload to Kaggle.  Uncomment to write when using the test.unlabeled data
with open('out.csv', 'w', newline = '') as results_csv:
    r_csv = csv.writer(results_csv, delimiter = ',')
    r_csv.writerow(["instance_id", "class"])
    for x in full_array:
        r_csv.writerow([x[0], x[1]])
results_csv.close()

###  Lines used to get the F1 score of the dev data
# score = f1_score(y_true, y_pred, average = 'weighted')
# print(score)

### Used to create a plot of f1_score and smoothing_alpha
def plotting(r, inst):
    y_true = []
    y_pred = []
    full_array = []
    for x2 in inst:
        x = np.array(classify(x2, pH, pNotH, smoothing_alpha = r, testing = True))
        full_array.append(x)
        y_true.append(x[1])
        if x[2] == 'match':
            y_pred.append(x[1])
        else:
            if x[1] == '1':
                y_pred.append(0)
            else:
                y_pred.append(1)
    score = f1_score(y_true, y_pred, average = 'micro')
    return(r, score)

### Uncomment the below, set classify to false, and comment out both returns before 'if testing:' to create a plot of finding best smoothing_alpha.
# alpha1 = []
# sc1 = []
# for r in np.arange(0.1,1.01,0.01):
#     alpha1.append(plotting(r, z2)[0])
#     sc1.append(plotting(r, z2)[1])
#     print(plotting(r, z2))
# data = [go.Scatter(
#         x=alpha1,
#         y=sc1,
#         textposition = 'auto',
#         marker=dict(
#             color='rgb(255,140,0)',
#             line=dict(
#                 color='rgb(8,48,107)',
#                 width=1.5),
#         ),
#         opacity=0.8
#     )]
# layout = go.Layout(
#         title = 'F1 Score as Smoothing Alpha Increases',
#         xaxis=dict(
#             tickangle=45,
#             tickfont=dict(
#                 size=10,
#                 color='rgb(107, 107, 107)'
#             )
#         ),
#         yaxis=dict(
#             title='F1 Score',
#             titlefont=dict(
#                 size=14,
#                 color='rgb(107, 107, 107)'
#             ),
#             tickfont=dict(
#                 size=12,
#                 color='rgb(107, 107, 107)'
#             )
#         )
#     )
# print('opening...')
# fig = go.Figure(data = data, layout = layout)
# py.offline.plot(fig, filename="bayes_plot.html")
