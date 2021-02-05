#Naive Bayes Classification
from sklearn.naive_bayes import GaussianNB
import joblib

joblib_file = "NaiveBayes_model.pkl"

def NaiveBayes(x_train,x_test,y_train):
    #create a Naive Bayes model
    model = GaussianNB()
    #fit model
    model.fit(x_train, y_train)
    #prediction of test data
    y_pred = model.predict(x_test)
    #save model with joblib
    #joblib.dump(model, joblib_file)
    return y_pred
