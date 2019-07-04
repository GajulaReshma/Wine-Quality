# coding: utf-8
# NOTE THIS SCRIPT IS WRITTEN FOR python2.7

import pandas as pd
import numpy as np

# Assumes inputs are pandas data frames
# Assumes the last column of data is the output dimension

##############################
# PART 1
print '\n ############ PART 1 ############# \n'
##############################

# Logistic Regression
# Assumes the last column of data is the output dimension
def get_pred_logreg(train,test):
    # Your implementation goes here
    # You may leverage the linear_model module from sklearn (scikit-learn)
    # return (predicted output, actual output)
    return None

# Support Vector Machine
# Assumes the last column of data is the output dimension
def get_pred_svm(train,test):
    # Your implementation goes here
    # You may leverage the svm module from sklearn (scikit-learn)
    # return (predicted output, actual output)
    return None

# Naive Bayes
# Assumes the last column of data is the output dimension
def get_pred_nb(train,test):
    # Your implementation goes here
    # You may leverage the naive_bayes module from sklearn (scikit-learn)
    # return (predicted output, actual output)
    return None

# k-Nearest Neighbor
# Assumes the last column of data is the output dimension
# Hint: you might want to use predit_proba to get predicted probabilities
def get_pred_knn(train,test,k):
    # Your implementation goes here
    # You may leverage the neighbors module from sklearn (scikit-learn)
    # return (predicted output, actual output)
    return None


##############################
# PART 2
print '\n ############ PART 2 ############# \n'
##############################

#your implementation of do_cv_class goes here
def do_cv_class(df, num_folds, model_name):
    return None

##############################
# PART 3
print '\n ############ PART 3 ############# \n'
##############################

#input prediction file the first column of which is prediction value
#the 2nd column is true label (0/1)
#cutoff is a numeric value, default is 0.5
def get_metrics(pred, cutoff=0.5):
    ### your implementation goes here
    return None

####################
####import data#####
print '\n ############ Import data ############# \n'
####################

my_data = pd.read_csv('wine.csv')
#encode class into 0/1 for easier handling by classification algorithm
my_data['type'] = np.where(my_data['type'] == 'high', 1, 0)

# scroll down to refer test cases below


##############################
# PART 4
print '\n ############ PART 4 ############# \n'
##############################


#test cases for "do_cv_class" and "get_metrics" functions
'''
print '-------------------'
print 'logistic regression'
print '-------------------'
tmp = do_cv_class(my_data,10,'logreg') # returns pandas dataframe
print_cont_table(tmp.iloc[:, 0:2])
print get_metrics(tmp.iloc[:, 0:2])


print '--------------------'
print 'naieve Bayes'
print '--------------------'
tmp = do_cv_class(my_data,10,'nb') # returns pandas dataframe
print_cont_table(tmp.iloc[:, 0:2])
print get_metrics(tmp.iloc[:, 0:2])

print '--------------------'
print 'svm'
print '--------------------'
tmp = do_cv_class(my_data,10,'svm') # returns pandas dataframe
print_cont_table(tmp.iloc[:, 0:2])
print get_metrics(tmp.iloc[:, 0:2])

print '--------------------'
print 'knn'
print '--------------------'
print = do_cv_class(my_data,10,'7nn') # returns pandas dataframe
print_cont_table(tmp.iloc[:, 0:2])
print get_metrics(tmp.iloc[:, 0:2])
'''
