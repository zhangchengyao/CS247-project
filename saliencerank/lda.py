from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd

# read the title and the body from the .csv
df = pd.read_csv("data_rake.csv")
df = df[pd.notnull(df["Title"])]
df = df[pd.notnull(df["Body"])]
df = df[pd.notnull(df["Tags"])]
title_array = df["Title"].to_numpy()
body_array = df["Body"].to_numpy()
line_num = 1000 # number of line to read from .csv
# combine them and count the word from the combination 
comb_array = []
for i in range(line_num):
    comb_array.append(title_array[i] + '. ' + body_array[i])
# comb_array = [title_array[i] + '. ' + body_array[i] for i in range(len(title_array))]

# use LDA to generate the parameters from doc to topic
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(comb_array) # TODO: ?Do I need to turn it into array
words = vectorizer.get_feature_names()
'''
if ',' in words:
    print("There is a comma")
else:
    print("There is not")
'''
topic_num = 500
lda = LatentDirichletAllocation(n_components=topic_num, random_state=0)
lda.fit(X)
lda_doc2topic = lda.transform(X)

# write them into file
# get the doc2topic
doc2topic_file = open("lda-docxXtopics-data"+str(line_num)+".txt", mode = "w")
for i in range(len(lda_doc2topic)):
    # write a line
    line = [str(integer) for integer in lda_doc2topic[i]]
    line = ','.join(line)
    doc2topic_file.write(line+'\n')
doc2topic_file.close()
# get the topic2vocab
topic2vocab_file = open("lda-topicsXvocab-data"+str(line_num)+".txt", mode = "w")
for i in range(topic_num):
    # write a line: vocab + '?' + component
    line = [words[j] + '\t' + str(int(lda.components_[i][j])) for j in range(len(words))]
    line = ','.join(line)
    topic2vocab_file.write(line+'\n')
topic2vocab_file.close()