"""
Import the DecisionTreeClassifier model.
"""
#Import the DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
###########################################################################################################
##########################################################################################################
"""
Import the Zoo Dataset
"""
#Import the dataset 
dataset = pd.read_csv('data/zoo.csv')
###########################################################################################################
##########################################################################################################
"""
Split the data into a training and a testing set
"""
train_features = dataset.iloc[:80,:-1]
test_features = dataset.iloc[80:,:-1]
train_targets = dataset.iloc[:80,-1]
test_targets = dataset.iloc[80:,-1]
###########################################################################################################
##########################################################################################################
"""
Train the model
"""
tree = DecisionTreeClassifier(criterion = 'entropy').fit(train_features,train_targets)
###########################################################################################################
##########################################################################################################
"""
Predict the classes of new, unseen data
"""
prediction = tree.predict(test_features)
###########################################################################################################
##########################################################################################################
"""
Check the accuracy
"""
print("The prediction accuracy is: ",tree.score(test_features,test_targets)*100,"%")