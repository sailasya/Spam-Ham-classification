#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import pandas as pd
import math
import string
import sys


# In[13]:


def extract_from_file(dirName,className) :
    message_list = list()
    for filename in os.listdir(dirName):
            with open(os.path.join(dirName, filename),encoding='utf8',errors='replace') as f:
                file_content = f.read().replace('\n',' ')
                message =[str(file_content),className]
                message_list.append(message)
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
            if unique_words[i] in row['Document']:
                data[i] = row['Document'].count(unique_words[i])
        processed_data_set.loc[idx] = data
    return processed_data_set.apply(pd.to_numeric)
            
def train_multinomial_nb(data,class_label):
    n = data.shape[0]
    v = data.iloc[:, :-1].columns
    no_of_vocab = len(v)
    condprob = dict()
    prior = dict()
    for c in class_label:
        nc = (len(data[(data['Class'] == c)]))
        prior[c] = math.log(nc,2)-math.log(n,2)
        tct = list()
        for t in v:
            tct.append(data.loc[data['Class'] == c, t].sum())
        sum_of_tct = sum(tct)
        i=0
        for t in v:
            if t in condprob:
                condprob[t][c] = math.log(tct[i]+1,2)-math.log(sum_of_tct+no_of_vocab,2)
            else:
                condprob[t] = {c:(math.log(tct[i]+1,2)-math.log(sum_of_tct+no_of_vocab,2))}
            i = i+1
    return v,prior,condprob

def test_multinomial_nb(class_label,v,prior,cond_prob,data,train_data):
    pred = list()
    ham_label = str(class_label[0])
    spam_label  = str(class_label[1])
    ham_data = train_data[train_data['Class']==ham_label]
    spam_data = train_data[train_data['Class']==spam_label]
    spam_word_count = spam_data.iloc[:,:-1].sum(axis = 1).sum()
    ham_word_count = ham_data.iloc[:,:-1].sum(axis = 1).sum()

    prob_ham = prior[ham_label]
    prob_spam = prior[spam_label]
    for index, row in data.iterrows():
        unique_words = set(row['Document'].split())
        prob_spam_email = prob_spam
        prob_ham_email = prob_ham
        for w in unique_words:
            if w in v :
                pr_WS = cond_prob[w][spam_label]
                pr_WH = cond_prob[w][ham_label]
                pr_SW = pr_WS
                pr_HW = pr_WH
            else:
                pr_WS = -math.log(train_data.shape[1]-1+spam_word_count,2)
                pr_WH = -math.log(train_data.shape[1]-1+ham_word_count,2)
                pr_SW = pr_WS
                pr_HW = pr_WH
            prob_spam_email+=pr_SW
            prob_ham_email+=pr_HW
        if prob_spam_email>=prob_ham_email :
            pred.append(spam_label)
        else :
            pred.append(ham_label)
    return pd.DataFrame(pred,columns=['Class'])


def calculate_performance_metrics(data,pred_class):
    true_positive,false_positive,true_negative,false_negative = 0,0,0,0
    for ind in data.index:
        if(data['Class'][ind]==str(0) and pred_class['Class'][ind]==str(0)):
            true_positive = true_positive + 1
        elif(data['Class'][ind]==str(1) and pred_class['Class'][ind]==str(0)):
            false_negative = false_negative + 1
        elif(data['Class'][ind]==str(0) and pred_class['Class'][ind]==str(1)):
            false_positive = false_positive + 1
        else:
            true_negative = true_negative + 1
    precision = true_positive/(true_positive + false_positive)
    accuracy = (true_positive + true_negative)/data.shape[0]
    recall  = true_positive/(true_positive + false_negative)
    F1 = 2*float(precision*recall)/(precision+recall)
    return accuracy,precision,recall,F1
            

def multinomial_nb(bow_train_data,bow_test_data,class_label):
    v,prior,condprob = train_multinomial_nb(bow_train_data,class_label)
    pred_class = test_multinomial_nb(class_label,v,prior,condprob,bow_test_data,bow_train_data)
    accuracy, precision, recall, F1 = calculate_performance_metrics(bow_test_data,pred_class)
    return accuracy, precision, recall, F1
    
def apply_multinomial_naive_bayes_model(file_path):
    class_label = ['0','1']
    ham_class = class_label[0]
    spam_class = class_label[1]
    
    train_data,test_data = extract_data(file_path,ham_class,spam_class)
    train_data,unique_words = preprocess_data(train_data)
    unique_words = list(unique_words)
    
    X_train = pd.DataFrame(train_data['Document'])
    Y_train = pd.DataFrame(train_data['Class'])
    X_train = process_data(X_train,unique_words)
    
    X_test,words = preprocess_data(test_data)
    X_test = pd.DataFrame(X_test,columns = ['Document'])
    Y_test = pd.DataFrame(test_data['Class'])
    
    X_train['Class'] = Y_train['Class'].values
    X_test['Class'] = Y_test['Class'].values
    
    bow_train_data,bow_test_data = X_train, X_test
    return multinomial_nb(bow_train_data,bow_test_data,class_label)


# In[14]:


if __name__ == '__main__':

    arg_list = sys.argv
    
    path =str(arg_list[1])
    accuracy, precision, recall,F1 = apply_multinomial_naive_bayes_model(path)
    print("Metrics in Multinomial Naive Bayes-Bag Of Words ")
    print("Accuracy:",accuracy*100)
    print("Precision:",precision*100)
    print("Recall:",recall*100)
    print("F1:",F1*100)
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




