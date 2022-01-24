#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
import pandas as pd
import numpy as np
import math
import random
import string
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
import sys


# In[21]:


def extract_from_file(dirName,className) :
    message_list = list()
    for filename in os.listdir(dirName):
            with open(os.path.join(dirName, filename),encoding='utf8',errors='replace') as f:
                file_content = f.read().replace('\n',' ')
                message =[str(file_content),className]
                message_list.append(message)
    random.seed(0)
    random.shuffle(message_list)
    return message_list

    
def extract_data(file_path,ham_class,spam_class):
    ham_train_data = extract_from_file(file_path +'/train/ham',ham_class)
    spam_train_data = extract_from_file(file_path+'/train/spam',spam_class)
    ham_test_data = extract_from_file(file_path+'/test/ham',ham_class)
    spam_test_data = extract_from_file(file_path+'/test/spam',spam_class)
    
    train_data = ham_train_data + spam_train_data
    test_data = ham_test_data + spam_test_data
    
    train_data = pd.DataFrame(train_data,columns=['Document','Class'])
    test_data = pd.DataFrame(test_data,columns=['Document','Class'])
    
    return train_data,test_data

def preprocess_data(data):
    x =string.printable
    waste = list(x)
    unique_words = list()
    unique_words.append('weight_zero')
    processed_data = []
    prepositions = ['and','the','or','are','in', 'to', 'be','is','as', 'by','if','will','as','for',
                    'on','it','we','than','this','an']
    for index, row in data.iterrows():
        words_list = row['Document'].split()
        words_list = [i for i in words_list if i not in waste]
        words_list = [i for i in words_list if i.isalnum() is True]
        words_list = [i for i in words_list if i not in prepositions]
        words_list = [i for i in words_list if len(i)!=2]
        unique_words.extend(words_list)
        data.loc[index,'Document'] = " ".join(words_list)
    
    return data,set(unique_words)

def process_data(dataset,unique_words):
    processed_data_set = pd.DataFrame(columns = unique_words)
    for idx,row in dataset.iterrows():
        data = [0]*len(unique_words)
        for i in range(len(unique_words)):
            if unique_words[i]=='weight_zero':
                data[i] = 1
            else:
                if unique_words[i] in row['Document']:
                    data[i] = 1
                else:
                    data[i] = 0
        processed_data_set.loc[idx] = data
    return processed_data_set.apply(pd.to_numeric)

def sigmoid(x):
    return (1/(1+np.exp(-x)))

def calculate_performance_metrics(class_y,pred_class):
    true_positive,false_positive,true_negative,false_negative = 0,0,0,0
    i = 0
    for idx,row in class_y.iterrows():
        if row['Class'] == pred_class[i]:
            if row['Class'] == '1':
                true_negative = true_negative + 1
            else:
                true_positive = true_positive + 1
        else:
            if row['Class'] == '1':
                false_negative = false_negative + 1
            else:
                false_positive = false_positive + 1
        i= i+1
    precision = true_positive/(true_positive + false_positive)
    accuracy = (true_positive + true_negative)/class_y.shape[0]
    recall  = true_positive/(true_positive + false_negative)
    F1 = 2*float(precision*recall)/(precision+recall)
    return accuracy,precision,recall,F1

def sgdclassifier(X_train,Y_train,X_test,Y_test):
    parameter_grid = {'alpha':[0.01,0.1,0.3]}
    sgd_classifier =SGDClassifier(random_state=0,loss='log',penalty='l2',class_weight='balanced',max_iter=1000)
    sgd_grid = GridSearchCV(estimator=sgd_classifier,param_grid=parameter_grid,n_jobs=-1,scoring='roc_auc')
    sgd_grid.fit(X_train,np.array(Y_train))
    pred_class = sgd_grid.predict(X_test)
    accuracy,precision,recall,F1 = calculate_performance_metrics(Y_test,pred_class)
    
    return accuracy,precision,recall,F1


def apply_bern_SGDClassifier_model(file_path):
    class_label = ['0','1']
    ham_class = class_label[0]
    spam_class = class_label[1]
    train_data,test_data = extract_data(file_path,ham_class,spam_class)
    
    X_train,train_unique_words = preprocess_data(train_data.loc[:, train_data.columns != 'Class'])
    train_unique_words = list(train_unique_words)
    X_train = process_data(X_train,train_unique_words)
    
    X_test,test_unique_words = preprocess_data(test_data.loc[:, test_data.columns != 'Class'])
    X_test = process_data(X_test,train_unique_words)
    
    
    Y_train = pd.DataFrame(train_data['Class'])
    Y_test = pd.DataFrame(test_data['Class'])
    
    return sgdclassifier(X_train,Y_train,X_test,Y_test)
    
    
    
    


# In[ ]:


if __name__ == '__main__':

    arg_list = sys.argv
    
    path =str(arg_list[1])
    accuracy, precision, recall,F1 = apply_bern_SGDClassifier_model(path)
    print("Metrics in SGDClassifier-Bernoullli ")
    print("Accuracy:",accuracy*100)
    print("Precision:",precision*100)
    print("Recall:",recall*100)
    print("F1:",F1*100)
    

