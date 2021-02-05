import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import re
from CrossValidation import *
from FeatureExtraction import *
from RandomForest import *
from SVM import *
from KNN import *
from ResamplingData import *
from Preprocessing import *
from DecisionTree import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from NaiveBayes import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


#split train and test data with bag of words features and data labels

x_train, x_test, y_train, y_test=CV(data_features,data)

print('Number of train samples:')
print(y_train.value_counts())
      
print('Number of test samples:')
print(y_test.value_counts())
      
#Classification Models

#1-RandomForest

print("_____________________________________")

print("Random Forest")

#Random Forest Classification
y_predict_R=randomForest(x_train,x_test,y_train)

#Classification accuracy
accuracy_R = round(accuracy_score(y_predict_R, y_test) * 100, 2)

#Print accuracy
print("Classification Accuracy: ",accuracy_R,'%')

#Confusion matrix
cf_1 = confusion_matrix(y_test, y_predict_R)

#Test data unique labels 
cf_labels=np.unique(y_test)

#creating data frame for confusion matrix
cf_1=pd.DataFrame(cf_1,index=cf_labels,columns=cf_labels)

#Precision: tp / (tp + fp)
precision=precision_score(y_test, y_predict_R, average="macro")

#Print precision result
print("Precision: ",precision)

#Recall: tp / (tp + fn)
recall=recall_score(y_test, y_predict_R, average="macro")

#Print recall result
print("Recall: ",recall)

#F1: 2 tp / (2 tp + fp + fn)
f1=f1_score(y_test, y_predict_R, average="macro")
#Print results for Random Forest
print("F1 score: ",f1)
print("__Confusion Matrix Of Random Forest__")
print("0:Erzincan, 1:Erzurum, 2:Sivas")
print(cf_1)

print("_____________________________________")


#2-NaiveBayes

print("Naive Bayes")

#Naive Bayes Classification
y_predict_N=NaiveBayes(x_train,x_test,y_train)

#Classification accuracy
accuracy_N = round(accuracy_score(y_predict_N, y_test) * 100, 2)

#Print accuracy
print("Classification Accuracy: ",accuracy_N,'%')

#Confusion matrix
cf_2=confusion_matrix(y_test, y_predict_N)

cf_2=pd.DataFrame(cf_2,index=cf_labels,columns=cf_labels)

#Precision tp / (tp + fp)
precision=precision_score(y_test, y_predict_N, average="macro")

print("Precision: ",precision)

#Recall tp / (tp + fn)
recall=recall_score(y_test, y_predict_N, average="macro")

print("Recall: ",recall)

#f1: 2 tp / (2 tp + fp + fn)
f1=f1_score(y_test, y_predict_N, average="macro")

print("F1 score: ",f1)
print("__Confusion Matrix Of Naive Bayes__")
print("0:Erzincan, 1:Erzurum, 2:Sivas")
print(cf_2)
print("_____________________________________")


#3-K-Nearest Neighbors

print("K-Nearest Neighbors")

#KNN classification
y_predict_K=KNN(x_train,x_test,y_train)

#accuray: (tp+tn)/(p+n)
accuracy_K = round(accuracy_score(y_predict_K, y_test) * 100, 2)

print("Classification Accuracy: ",accuracy_K,'%')

#confusion matrix
cf_3=confusion_matrix(y_test, y_predict_K)

cf_3=pd.DataFrame(cf_3,index=cf_labels,columns=cf_labels)

#precion tp / (tp + fp)
precision=precision_score(y_test, y_predict_K, average="macro")

print("Precision: ",precision)
#Recall tp / (tp + fn)
recall=recall_score(y_test, y_predict_K, average="macro")

print("Recall: ",recall)

#F1: 2 tp / (2 tp + fp + fn)
f1=f1_score(y_test, y_predict_K, average="macro")

print("F1 score: ",f1)
print("__Confusion Matrix Of KNN__")
print("0:Erzincan, 1:Erzurum, 2:Sivas")
print(cf_3)
print("_____________________________________")

#4-DecisionTree

print("Decision Tree")

#Decision Tree Classification
y_predict_D=decisionTree(x_train,x_test,y_train)

#accuracy 
accuracy_D = round(accuracy_score(y_predict_D, y_test) * 100, 2)

print("Classification Accuracy: ",accuracy_D,'%')

#confusion matrix
cf_4=confusion_matrix(y_test, y_predict_D)

cf_4=pd.DataFrame(cf_4,index=cf_labels,columns=cf_labels)

#precion tp / (tp + fp)
precision=precision_score(y_test, y_predict_D, average="macro")

print("Precision: ",precision)

#Recall tp / (tp + fn)
recall=recall_score(y_test, y_predict_D, average="macro")

print("Recall: ",recall)

#F1: 2 tp / (2 tp + fp + fn)
f1=f1_score(y_test, y_predict_D, average="macro")

print("F1 score: ",f1)
print("__Confusion Matrix Of Decision Tree__")
print("0:Erzincan, 1:Erzurum, 2:Sivas")
print(cf_4)
print("_____________________________________")

#5-Support Vector Machine

print("Support Vector Machine")

#SVM classification
y_predict_S=SVM(x_train,x_test,y_train)

accuracy_S = round(accuracy_score(y_predict_S, y_test) * 100, 2)

print("Classification Accuracy: ",accuracy_S,'%')
#confusion matrix
cf_5=confusion_matrix(y_test, y_predict_S)

cf_5=pd.DataFrame(cf_5,index=cf_labels,columns=cf_labels)

#precision tp / (tp + fp)
precision=precision_score(y_test, y_predict_S, average="macro")

print("Precision: ",precision)

#Recall tp / (tp + fn)
recall=recall_score(y_test, y_predict_S, average="macro")

print("Recall: ",recall)
#F1: 2 tp / (2 tp + fp + fn)

f1=f1_score(y_test, y_predict_S, average="macro")
print("F1 score: ",f1)

print("__Confusion Matrix Of SVM__")
print("0:Erzincan, 1:Erzurum, 2:Sivas")
print(cf_5)

print("_____________________________________")

