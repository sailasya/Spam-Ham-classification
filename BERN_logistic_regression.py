#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import math
import random
import string
import sys


# In[4]:


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

def split_into_train_and_validation(data):
    l = len(data)
    d = (0.7*l)
    train_data = []
    validation_data = []
    for i in range(l):
        if i>=d:
            validation_data.append(data[i])
        else:
            train_data.append(data[i])
    return train_data,validation_data
    
def extract_data(file_path,ham_class,spam_class):
    ham_train_data = extract_from_file(file_path +'/train/ham',ham_class)
    spam_train_data = extract_from_file(file_path+'/train/spam',spam_class)
    ham_test_data = extract_from_file(file_path+'/test/ham',ham_class)
    spam_test_data = extract_from_file(file_path+'/test/spam',spam_class)
    
    train_data = ham_train_data + spam_train_data
    test_data = ham_test_data + spam_test_data
    
    ham_train_data,ham_validation_data = split_into_train_and_validation(ham_train_data)
    spam_train_data,spam_validation_data = split_into_train_and_validation(spam_train_data)
    
    split_train_data = ham_train_data + spam_train_data
    split_validation_data = ham_validation_data + spam_validation_data
    
    split_train_data = pd.DataFrame(split_train_data,columns=['Document','Class'])
    test_data = pd.DataFrame(test_data,columns=['Document','Class'])
    split_validation_data = pd.DataFrame(split_validation_data,columns = ['Document','Class'])
    train_data = pd.DataFrame(train_data,columns=['Document','Class'])
    return split_train_data,test_data,split_validation_data,train_data

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

def calculate_performance_metrics(data,weights,class_y):
    true_positive,false_positive,true_negative,false_negative = 0,0,0,0
    pred_class = []
    pred_y = np.array(np.dot(data,weights),dtype=np.float32)
    for i in pred_y:
        if i>0:
            pred_class.append('1')
        else:
            pred_class.append('0')
    i=0
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
        i = i+1
    precision = true_positive/(true_positive + false_positive)
    accuracy = (true_positive + true_negative)/data.shape[0]
    recall  = true_positive/(true_positive + false_negative)
    F1 = 2*float(precision*recall)/(precision+recall)
    return accuracy,precision,recall,F1

def decide_lambda(datax,datay,valid_datax,valid_datay):
    lambda_var = [0.1,0.3,0.5,0.7,0.9]
    min_acc = 0
    final_lambda = lambda_var[0]
    for l in lambda_var:
        weights = np.zeros(datax.shape[1])
        for i in range(300):
            X = np.array(np.dot(datax,weights),dtype=np.float32)
            S = sigmoid(X)
            C = np.array(datay.apply(pd.to_numeric)).reshape(datay.shape[0])
            g = np.dot(datax.T,C-S)
            weights = weights + (0.01*g)-(0.01*l*weights)
        validation_data = valid_datax.values
        accuracy,precision,recall,F1 = calculate_performance_metrics(validation_data,weights,valid_datay)
        if accuracy>=min_acc:
            final_lambda = l
            min_acc = accuracy
    return final_lambda

def logistic_regression(X_train_data,Y_train_data,weights,final_lambda):
    for i in range(1000):
        x = np.array(np.dot(X_train_data,weights),dtype=np.float32)
        S = sigmoid(x)
        C = np.array(Y_train_data.apply(pd.to_numeric)).reshape(Y_train_data.shape[0])
        g = np.dot(X_train_data.T,C-S)
        weights = weights+(0.01*(g))-(0.01*final_lambda*weights)
    return weights

def apply_bern_logistic_regression_model(file_path):
    class_label = ['0','1']
    ham_class = class_label[0]
    spam_class = class_label[1]
    train_data,test_data,validation_data,total_train_data = extract_data(file_path,ham_class,spam_class)
    
    X_train,train_unique_words = preprocess_data(train_data.loc[:, train_data.columns != 'Class'])
    train_unique_words = list(train_unique_words)
    X_validation,validation_unique_words = preprocess_data(validation_data.loc[:, validation_data.columns != 'Class'])
    
    
    Y_train = pd.DataFrame(train_data['Class'])
    Y_test = pd.DataFrame(test_data['Class'])
    Y_validation = pd.DataFrame(validation_data['Class'])
    Y_total_train_data = pd.DataFrame(total_train_data['Class'])
    
    X_train = process_data(X_train,train_unique_words)
    X_validation = process_data(X_validation,train_unique_words)
    
    weights = np.zeros(X_train.shape[1])
    final_lambda = decide_lambda(X_train,Y_train,X_validation,Y_validation)
    print("Final lambda",final_lambda)
    
    X_total_train_data,total_train_unique_words = preprocess_data(total_train_data.loc[:, total_train_data.columns != 'Class'])
    total_train_unique_words = list(total_train_unique_words)
    X_total_train_data = process_data(X_total_train_data,total_train_unique_words)
    weights = np.zeros(X_total_train_data.shape[1])
    weights = logistic_regression(X_total_train_data,Y_total_train_data,weights,final_lambda)
    
    X_test,test_unique_words = preprocess_data(test_data.loc[:, test_data.columns != 'Class'])
    X_test = process_data(X_test,total_train_unique_words)
    test_data = X_test.values
    accuracy,precision,recall,F1 = calculate_performance_metrics(test_data,weights,Y_test)
    
    return accuracy,precision,recall,F1
    


# In[ ]:


if __name__ == '__main__':

    arg_list = sys.argv
    
    path =str(arg_list[1])
    accuracy, precision, recall,F1 = apply_bern_logistic_regression_model(path)
    print("Metrics in Logistic Regression-Bernoulli")
    print("Accuracy:",accuracy*100)
    print("Precision:",precision*100)
    print("Recall:",recall*100)
    print("F1:",F1*100)

