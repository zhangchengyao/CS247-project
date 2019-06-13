#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Code for processing the datasets, writing the extracted keyphrases 
to files and getting the final accuracy statistics. 
'''

import os, io, sys
from ranks import saliencerank, textrank, tpr, singletpr
from utils import * 
import numpy as np 
import pandas as pd
import time

"""Outputs the keyphrases in sepatare text files."""
def writeFiles(keyphrases, fileName, fileDir):
    keyphraseFile = io.open(fileDir + "/" + fileName + ".txt", 'wb')
    keyphraseFile.write('; '.join(keyphrases))
    keyphraseFile.close()


"""Switch for running the different algorithms"""
def algorithm_switch (argument, topics, pt, txt, article_ID, alpha=0.1):
    if argument == 0: 
        return textrank(txt) 
    elif argument == 1: 
        return tpr (topics, pt, txt, article_ID)
    elif argument == 2: 
        return saliencerank (topics, pt, txt, article_ID, alpha)
    elif argument == 3: 
        return singletpr (topics, pt, txt, article_ID)

"""Process the Inspec (Hulth2003) dataset
    The keyphrases for each document are written to files. """
def process_hulth (lda_file, docsXtopics_file, output_dir, flag):
    directory = "data/inspec/all"
    articles = os.listdir(directory)
    
    text_articles = []
    for article in articles: 
        if article.endswith (".abstr"): 
            text_articles.append(article)
    text_articles.sort()
    
    pt = load_docsXtopics_from_file (docsXtopics_file)
    pt = np.array(pt, dtype = 'float64')
    topics = parse_weights_from_file (lda_file)
    
    for article_ID in xrange (len(text_articles)):
        articleFile = io.open(directory + "/" + text_articles[article_ID], 'rb')
        text = articleFile.read()
        text=text.strip('\t\n\r')
        text= text.split('\r\n') 

        phrases = algorithm_switch (flag, topics, pt, text[1], article_ID)
        print(phrases)
        phrases_topk = []
        for k, _ in phrases:
            phrases_topk.append(k)
        writeFiles(phrases_topk, text_articles[article_ID], output_dir)


"""Process the 500N dataset
    The keyphrases for each document are written to files. """
def process_500N (lda_file, docsXtopics_file, output_dir, flag):
    directory = "data/500N/all"
    
    articles = os.listdir(directory)
    
    text_articles = []
    for article in articles: 
        if article.endswith (".txt"): 
            text_articles.append(article)
    text_articles.sort()
    
    pt = load_docsXtopics_from_file (docsXtopics_file)
    pt = np.array(pt, dtype = 'float64')
    topics = parse_weights_from_file (lda_file)
    
    for article_ID in xrange (len(text_articles)):
        articleFile = io.open(directory + "/" + text_articles[article_ID], 'rb')
        text = articleFile.read()
        text= text.split('\n') 
        
        phrases = algorithm_switch (flag, topics, pt, text[1], article_ID)
        phrases_topk = []
        for k, _ in phrases:
            phrases_topk.append(k)
        writeFiles(phrases_topk, text_articles[article_ID], output_dir)

def process_data(lda_file, docsXtopics_file, output_dir, flag, line_num):
    df = pd.read_csv("data_rake.csv")
    df = df[pd.notnull(df["Title"])]
    df = df[pd.notnull(df["Body"])]
    df = df[pd.notnull(df["Tags"])]
    title_array = df["Title"].to_numpy()
    body_array = df["Body"].to_numpy()
    tag_array = df["Tags"].to_numpy()
    # combine them and count the word from the combination 
    comb_array = []
    for i in range(line_num):
        comb_array.append(title_array[i] + '. ' + body_array[i])
    
    pt = load_docsXtopics_from_file (docsXtopics_file)
    pt = np.array(pt, dtype = 'float64')
    topics = parse_weights_from_file (lda_file)
    total_f1 = 0
    count = 1
    start_time = time.time()
    out_file = open("result_include_salience.txt", mode = "w")

    for question_ID,text in enumerate(comb_array):
        tp = fp = fn = 0
        phrases = algorithm_switch (flag, topics, pt, text, question_ID)
        out_file.write("phrase: "+str(phrases)+"\n")
        phrases_topk = [] 
        tags = tag_array[question_ID].split()
        tags = [tag.replace("-", " ") for tag in tags]
        out_file.write("tags: "+str(tags)+"\n")
        cand_set = set()
        # compute the f1-score
        for k, _ in phrases:
            phrases_topk.append(k)
            got_flag = False
            word_list = k.split()
            cand_set = cand_set|set(word_list)
            for word in word_list:
                if word in tags:
                    tp += 1
                    break
            if got_flag == False:
                fp += 1
        for k in tags:
            if k not in cand_set:
                fn += 1
        if tp==0:
            temp_f1 = 0
        else:
            precision = float(tp)/float(tp+fp)
            recall = float(tp)/float((tp+fn))
            temp_f1 = 2*float(precision*recall)/float(precision+recall)
        out_file.write("document_f1: "+str(temp_f1)+"\n")
        total_f1 += temp_f1
        # timer
        if count % 10 == 0:
            time_passed = (time.time()-start_time)/60
            print('{} reads aligned'.format(count), 'in {:.3} minutes'.format(time_passed))
            remaining_time = time_passed/count*(len(comb_array)-count)
            print('Approximately {:.3} minutes remaining'.format(remaining_time))
        count += 1
    avrg_f1 = total_f1/len(comb_array)
    out_file.write("")
    out_file.write("Average F1-score:"+ str(avrg_f1))
    out_file.close()

'''Runs a single algorithm on both Inspec (Hulth2003) and 500N datasets and outputs stats. '''
def process_datasets(algorithm): 

    output_dir = "output"
    line_num = 1000
    lda_file = "lda-topicsXvocab-data"+str(line_num)+".txt"
    docXtopics_file = "lda-docxXtopics-data"+str(line_num)+".txt"
    process_data(lda_file,docXtopics_file, output_dir,algorithm, line_num)
'''
    output_dir = "results/inspec"
    lda_file = "lda/lda-topicsXvocab-500-Hulth2003.txt"
    docXtopics_file = "lda/lda-docxXtopics-500-Hulth2003.txt"
    gold_standard_directory = "data/inspec/all/"
    process_hulth (lda_file,docXtopics_file, output_dir, algorithm)
    

    output_dir = "results/500N"
    lda_file = "lda/lda-topicsXvocab-500-500N.txt"
    docXtopics_file = "lda/lda-docxXtopics-500-500N.txt"
    gold_standard_directory = "data/500N/all/"
    process_500N(lda_file,docXtopics_file, output_dir, algorithm)
'''